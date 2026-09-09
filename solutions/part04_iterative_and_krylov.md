# Solutions: Part 4, Iterative and Krylov Methods

Worked solutions for the exercises in lessons 23 to 28.

Levels 1 and 2 are answered in full. Levels 3 and 4 give the method, the key code, and the
result you should get, so you can check your own work rather than copy it. Level 5 questions
are open ended, so those get a route through the problem and the answer where there is a
definite one.

Every number quoted was measured by running the code, not estimated.

---

## Lesson 23, Classical Iterative Methods

### 1.1 Why Jacobi parallelises and Gauss-Seidel does not

Jacobi computes every component of $\mathbf{x}^{(k+1)}$ from $\mathbf{x}^{(k)}$ alone:

$$x_i^{(k+1)} = \frac{1}{a_{ii}}\Big(b_i - \sum_{j \ne i} a_{ij}x_j^{(k)}\Big).$$

Nothing on the right depends on anything computed in this sweep, so all $n$ components are
independent and can be computed at the same time on $n$ processors. The sweep is exactly one
matrix-vector product plus a divide.

Gauss-Seidel uses the new values as soon as they exist:

$$x_i^{(k+1)} = \frac{1}{a_{ii}}\Big(b_i - \sum_{j<i} a_{ij}x_j^{(k+1)} - \sum_{j>i} a_{ij}x_j^{(k)}\Big).$$

Component $i$ needs component $i-1$ of the **same** sweep whenever $a_{i,i-1} \ne 0$, so there
is a chain of dependencies running the length of the sweep. That chain is the whole sweep, so
there is nothing to run in parallel.

**The dependency comes from the sparsity pattern, not from the method**, which is why red-black
ordering (exercise 3.2) recovers the parallelism: on a five point stencil no red point touches
another red point, so all red points are independent of each other.

### 1.2 Sweeps to gain ten digits at $\rho = 0.999$

The error shrinks by $\rho$ per sweep, so after $k$ sweeps it is $\rho^k$ times the start.
Gaining ten digits means $\rho^k = 10^{-10}$, hence

$$k = \frac{-10}{\log_{10}\rho} = \frac{-10}{\log_{10}0.999} = \frac{10}{4.3451\times10^{-4}}
\approx 23{,}013.$$

`nalib.iterative.iterations_needed(0.999, 1e-10)` returns 23012.8.

The useful shorthand is that the **digits gained per sweep** is $-\log_{10}\rho$, which for
$\rho$ close to 1 is about $(1-\rho)/\ln 10 \approx 0.434(1-\rho)$. So the cost is
$\approx 23/(1-\rho)$ sweeps per ten digits, and everything about a stationary method reduces
to how close $\rho$ is to 1.

### 1.3 Why Jacobi's iteration count grows like $n^2$

For the second difference matrix the Jacobi eigenvalues are $\lambda_k = \cos(k\pi h)$ with
$h = 1/(n+1)$, so

$$\rho_{\text{J}} = \cos(\pi h) = 1 - \frac{\pi^2h^2}{2} + O(h^4).$$

By 1.2 the sweep count is proportional to $1/(1-\rho) = 2/(\pi^2h^2) = O(h^{-2}) = O(n^2)$.

**The chain is worth stating as a chain**, because every step of it recurs: the discretisation
sets $h$, $h$ sets the eigenvalues, the extreme eigenvalue sets $\rho$, and $\rho$ sets the
count. Refining the grid to get a better answer makes the solver quadratically slower, so the
two goals fight each other. That is exactly the difficulty lesson 28 removes.

### 2.1 Proof of Theorem 23.1

**Claim.** $\mathbf{x}^{(k)} \to \mathbf{x}^\ast$ for every starting vector if and only if
$\rho(G) < 1$.

The error satisfies $\mathbf{e}^{(k)} = G^k\mathbf{e}^{(0)}$, so the statement is that
$G^k\mathbf{e} \to \mathbf{0}$ for every $\mathbf{e}$, which is the same as $G^k \to 0$.

**If $\rho(G) < 1$.** Pick $\varepsilon$ with $\rho(G) + \varepsilon < 1$. There is a norm with
$\|G\| \le \rho(G) + \varepsilon$: take the Jordan form $G = SJS^{-1}$, scale the Jordan blocks
by $D_\delta = \operatorname{diag}(1, \delta, \delta^2, \dots)$ so the off-diagonal ones become
$\delta$, and use $\|\mathbf{x}\| = \|D_\delta^{-1}S^{-1}\mathbf{x}\|_\infty$. Then
$\|G^k\| \le \|G\|^k \to 0$, and since all norms on $\mathbb{R}^n$ are equivalent (lesson 15),
$G^k \to 0$ in every norm.

**If $\rho(G) \ge 1$.** Let $\lambda$ be an eigenvalue with $|\lambda| \ge 1$ and
$\mathbf{v}$ its eigenvector. Then $G^k\mathbf{v} = \lambda^k\mathbf{v}$, whose norm is
$|\lambda|^k\|\mathbf{v}\|$, which does not go to zero. So starting from
$\mathbf{x}^{(0)} = \mathbf{x}^\ast + \mathbf{v}$ the iteration fails to converge.

**The "for every starting vector" is not decoration.** With $\rho(G) \ge 1$ the iteration still
converges from any $\mathbf{x}^{(0)}$ whose error happens to lie in the invariant subspace of
the small eigenvalues. That set has measure zero, and roundoff pushes you off it after one
sweep, so it is not a practical exception. But the theorem is false without the quantifier.

If $\lambda$ is complex, $\mathbf{v}$ is complex, and the real starting vector to use is
$\operatorname{Re}\mathbf{v}$ or $\operatorname{Im}\mathbf{v}$, at least one of which is
nonzero and still fails to decay.

### 2.2 Strict diagonal dominance implies Jacobi converges

Strict diagonal dominance means $|a_{ii}| > \sum_{j \ne i}|a_{ij}|$ for every $i$. Jacobi's
iteration matrix is $G = -D^{-1}(L+U)$, whose entries are

$$g_{ij} = \begin{cases} -a_{ij}/a_{ii} & j \ne i \\ 0 & j = i.\end{cases}$$

The infinity norm is the largest absolute row sum:

$$\|G\|_\infty = \max_i \sum_{j\ne i}\frac{|a_{ij}|}{|a_{ii}|}
= \max_i \frac{1}{|a_{ii}|}\sum_{j\ne i}|a_{ij}| < 1$$

by dominance. Since $\rho(G) \le \|G\|$ for any induced norm (lesson 15), $\rho(G) < 1$ and
Theorem 23.1 gives convergence.

**Note which direction this runs.** Dominance is sufficient and far from necessary. The second
difference matrix is only weakly dominant, with equality on the interior rows, and Jacobi
converges on it anyway with $\rho = \cos(\pi h) < 1$. The norm bound just is not sharp enough
to see it.

### 2.3 Kahan's theorem

**Claim.** $\rho(G_{\text{SOR}}) \ge |\omega - 1|$ for every $\omega$, so SOR can converge only
for $0 < \omega < 2$.

The determinant of a matrix is the product of its eigenvalues, so
$\rho(G)^n \ge |\det G|$. For SOR,

$$G_{\text{SOR}} = \left(\tfrac{D}{\omega} + L\right)^{-1}
\left(\tfrac{1-\omega}{\omega}D - U\right).$$

Both factors are triangular, so both determinants are products of diagonal entries:

$$\det\left(\tfrac{D}{\omega}+L\right) = \prod_i \frac{a_{ii}}{\omega} = \frac{\det D}{\omega^n},
\qquad
\det\left(\tfrac{1-\omega}{\omega}D - U\right) = \frac{(1-\omega)^n}{\omega^n}\det D.$$

Therefore $\det G_{\text{SOR}} = (1-\omega)^n$ and

$$\rho(G_{\text{SOR}}) \ge |\det G_{\text{SOR}}|^{1/n} = |1-\omega|.$$

For $\rho < 1$ we need $|1-\omega| < 1$, that is $0 < \omega < 2$.

**The result is remarkable for what it does not use.** No symmetry, no positive definiteness,
no structure at all beyond a nonzero diagonal. The bound comes from the determinant alone, and
the determinant is available because both triangles are triangular.

The bound is attained: at the optimal $\omega$ on the model problem,
$\rho(G_{\text{SOR}}(\omega^\ast)) = \omega^\ast - 1$ **exactly**, verified in lesson 23
section 5 at every size. So Kahan's inequality is not only a constraint, it tells you the best
you can hope for.

### 2.4 Gauss-Seidel converges for every SPD matrix

Write $A = D + L + L^T$ with $A$ symmetric positive definite, so $M = D + L$ and
$N = M - A = -L^T$. Let $\mathbf{e}$ be any nonzero error and
$\hat{\mathbf{e}} = G\mathbf{e}$ the error after one sweep, where $G = M^{-1}N$.

Set $\mathbf{d} = \hat{\mathbf{e}} - \mathbf{e}$. From $M\hat{\mathbf{e}} = N\mathbf{e}
= (M - A)\mathbf{e}$ we get $M\mathbf{d} = -A\mathbf{e}$.

Now expand the energy in terms of $\mathbf{d}$:

$$\hat{\mathbf{e}}^TA\hat{\mathbf{e}} - \mathbf{e}^TA\mathbf{e}
= 2\mathbf{d}^TA\mathbf{e} + \mathbf{d}^TA\mathbf{d}
= -2\mathbf{d}^TM\mathbf{d} + \mathbf{d}^TA\mathbf{d}
= -\mathbf{d}^T(2M - A)\mathbf{d}.$$

For Gauss-Seidel, $2M - A = 2(D+L) - (D+L+L^T) = D + L - L^T$, and since $L - L^T$ is
skew symmetric it contributes nothing to a quadratic form:

$$\mathbf{d}^T(2M-A)\mathbf{d} = \mathbf{d}^TD\mathbf{d} = \sum_i a_{ii}d_i^2 > 0$$

whenever $\mathbf{d} \ne \mathbf{0}$, because the diagonal of an SPD matrix is positive
($a_{ii} = \mathbf{e}_i^TA\mathbf{e}_i > 0$).

So the energy strictly decreases unless $\mathbf{d} = \mathbf{0}$, and $\mathbf{d} = \mathbf{0}$
forces $A\mathbf{e} = \mathbf{0}$, hence $\mathbf{e} = \mathbf{0}$. The energy
$\|\mathbf{e}\|_A^2$ is therefore a strictly decreasing function of a nonzero error, which by
compactness of the unit $A$-sphere gives $\|G\|_A < 1$ and so $\rho(G) < 1$.

**This is the same argument SOR needs**, with $2M - A = (2/\omega - 1)D + L - L^T$, positive
definite exactly when $\omega < 2$. Kahan's theorem and this one meet at $\omega = 2$ from
opposite sides.

### 2.5 Deriving $\rho_J = \cos(\pi h)$

Lesson 21 gives the eigenvalues of the second difference matrix
$A = \operatorname{tridiag}(-1, 2, -1)$ of size $n$:

$$\mu_k = 2 - 2\cos(k\pi h) = 4\sin^2\!\left(\frac{k\pi h}{2}\right), \qquad k = 1,\dots,n,$$

with $h = 1/(n+1)$. Jacobi's iteration matrix is $G_{\text{J}} = I - D^{-1}A = I - A/2$, since
$D = 2I$. So $G_{\text{J}}$ has the same eigenvectors and eigenvalues

$$\lambda_k = 1 - \frac{\mu_k}{2} = \cos(k\pi h).$$

The largest in absolute value is at $k = 1$ or $k = n$, and $\cos(n\pi h) = \cos(\pi - \pi h)
= -\cos(\pi h)$, so both give

$$\rho_{\text{J}} = \cos(\pi h).$$

**The $O(h^2)$ statement.** $\cos(\pi h) = 1 - \pi^2h^2/2 + O(h^4)$, so $1 - \rho_{\text{J}} =
\pi^2h^2/2 + O(h^4)$ and the sweep count is $O(h^{-2})$.

**The $O(h)$ statement for SOR.** With $\omega^\ast = 2/(1 + \sqrt{1-\rho_{\text{J}}^2})$ and
$\sqrt{1-\cos^2(\pi h)} = \sin(\pi h)$,

$$\omega^\ast = \frac{2}{1+\sin(\pi h)}, \qquad
\rho(\omega^\ast) = \omega^\ast - 1 = \frac{1-\sin(\pi h)}{1+\sin(\pi h)}.$$

Then $1 - \rho(\omega^\ast) = 2\sin(\pi h)/(1+\sin(\pi h)) \approx 2\pi h$, which is $O(h)$, so
the sweep count is $O(h^{-1}) = O(n)$.

**The whole gain is one power of $h$**, obtained by moving $\rho$ from $1 - O(h^2)$ to
$1 - O(h)$, and it required knowing $\rho_{\text{J}}$ in advance. Conjugate gradient (lesson 24)
gets the same power for free, without a parameter.

### 3.1 Matrix-free Jacobi and Gauss-Seidel

Jacobi is easy matrix-free, because a sweep is exactly one application of $A$:

$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + D^{-1}(\mathbf{b} - A\mathbf{x}^{(k)}).$$

So the operator interface needs only `apply(x)` and the diagonal:

```python
def jacobi_free(apply_A, diag, b, x0=None, tol=1e-10, max_iter=10000):
    """Jacobi given only a function computing A @ x and the diagonal of A."""
    b = np.asarray(b, dtype=float).ravel()
    d = np.asarray(diag, dtype=float).ravel()
    if d.size != b.size:
        raise ValueError(f"diagonal has length {d.size}, b has length {b.size}")
    if np.any(d == 0.0):
        raise np.linalg.LinAlgError("Jacobi needs a nonzero diagonal")
    x = np.zeros(b.size) if x0 is None else np.asarray(x0, dtype=float).ravel().copy()
    scale = float(np.linalg.norm(b)) or 1.0
    for k in range(max_iter):
        r = b - apply_A(x)
        if np.linalg.norm(r) <= tol * scale:
            return x, k
        x = x + r / d
    return x, max_iter
```

For the second difference operator with no matrix stored:

```python
def apply_second_difference(x):
    """(A x)_i = -x_{i-1} + 2 x_i - x_{i+1}, with zero outside. Size comes from x."""
    y = 2.0 * x
    y[:-1] -= x[1:]
    y[1:] -= x[:-1]
    return y
```

Both the operator and the solver take their size from the input vector, so the same code runs
at $n = 10$ and $n = 10^6$.

**Gauss-Seidel is the hard one**, and the difficulty is the point. A sweep needs the
**individual rows** in order, not the whole product, so `apply_A` alone is not enough. You need
either a row-access function `apply_row(i, x)` or, for a stencil, the update written out:

```python
def gs_free_second_difference(b, x, sweeps=1):
    """One Gauss-Seidel sweep for the second difference operator, no matrix stored."""
    n = x.size
    for _ in range(sweeps):
        for i in range(n):
            left = x[i-1] if i > 0 else 0.0
            right = x[i+1] if i + 1 < n else 0.0
            x[i] = (b[i] + left + right) / 2.0
    return x
```

**This asymmetry is a real practical fact.** Matrix-free codes routinely use Jacobi, Chebyshev
or polynomial smoothers precisely because they need only $A\mathbf{x}$, while Gauss-Seidel
needs a structure the operator interface deliberately hides. It is one more reason lesson 28
uses damped Jacobi.

### 3.2 Red-black Gauss-Seidel

Colour the unknowns so that no two neighbours share a colour. For the second difference
operator, alternate indices work: red is even, black is odd. Then every red update touches only
black values and vice versa, so each colour can be swept fully in parallel.

```python
def red_black_gauss_seidel(A, b, tol=1e-10, max_iter=10000):
    """Gauss-Seidel with all even indices updated first, then all odd ones.

    Sizes come from A, and the colouring is derived from the index parity, so this works at
    any n. For a general sparsity pattern replace the parity by a graph colouring.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    b = np.asarray(b, dtype=float).ravel()
    d = np.diag(A).copy()
    R = A - np.diag(d)
    colours = [np.arange(0, n, 2), np.arange(1, n, 2)]
    x = np.zeros(n)
    scale = float(np.linalg.norm(b)) or 1.0
    for k in range(1, max_iter + 1):
        for idx in colours:
            x[idx] = (b[idx] - R[idx] @ x) / d[idx]     # whole colour at once
        if np.linalg.norm(b - A @ x) <= tol * scale:
            return x, k
    return x, max_iter
```

The inner line updates a whole colour in one vector operation, which is what makes it
parallel.

**Measured on the second difference matrix**, against ordinary Gauss-Seidel to a relative
residual of $10^{-10}$:

| $n$ | natural order | red-black |
|---|---|---|
| 20 | 905 | 920 |
| 40 | 3285 | 3342 |
| 80 | 12145 | 12375 |

Red-black costs about **2 percent more sweeps** and is fully parallel, which is an
overwhelmingly good trade. The rates match because both orderings are consistent in the sense
of exercise 5.1, so Theorem 23.4 applies to both and gives
$\rho_{\text{GS}} = \rho_{\text{J}}^2$ for each.

### 3.3 Adaptive SOR

The optimal $\omega$ needs $\rho_{\text{J}}$, which is usually unknown. But the iteration
**reveals** its own rate: the ratio of successive residual norms tends to $\rho$. So estimate
$\rho$ as you go, invert the relation, and update $\omega$.

For a consistently ordered matrix, $\rho(G_{\text{SOR}}(\omega))$ relates to $\rho_{\text{J}}$
by $(\lambda + \omega - 1)^2 = \lambda\omega^2\rho_{\text{J}}^2$. At $\omega = 1$ this gives
$\rho_{\text{GS}} = \rho_{\text{J}}^2$, so the simplest scheme is:

```python
def adaptive_sor(A, b, tol=1e-10, max_iter=10000, warmup=10, recheck=25):
    """SOR that estimates rho_J from its own convergence and updates omega.

    Starts at omega = 1 (plain Gauss-Seidel), measures the observed rate over a window, infers
    rho_J from rho_GS = rho_J^2, and switches to the optimal omega. Sizes come from A.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    b = np.asarray(b, dtype=float).ravel()
    d = np.diag(A).copy()
    x = np.zeros(n)
    scale = float(np.linalg.norm(b)) or 1.0
    omega, res = 1.0, [float(np.linalg.norm(b))]
    for k in range(1, max_iter + 1):
        for i in range(n):
            gs = (b[i] - A[i, :i] @ x[:i] - A[i, i+1:] @ x[i+1:]) / d[i]
            x[i] = (1.0 - omega) * x[i] + omega * gs
        res.append(float(np.linalg.norm(b - A @ x)))
        if res[-1] <= tol * scale:
            return x, k, omega
        if omega == 1.0 and k >= 2 * warmup and res[-warmup] > 0:
            rate = (res[-1] / res[-warmup]) ** (1.0 / warmup)      # this is rho_GS
            rho_j = min(np.sqrt(max(rate, 0.0)), 1.0 - 1e-14)
            omega = float(np.clip(2.0 / (1.0 + np.sqrt(1.0 - rho_j**2)), 1.0, 1.99999))
    return x, max_iter, omega
```

**Estimate once, then stop.** An earlier version that kept re-estimating after the switch drove
$\omega$ straight to the clip at 1.99 and took four times as many sweeps, because inverting the
rate relation for $\omega \ne 1$ is far less stable than the clean
$\rho_{\text{GS}} = \rho_{\text{J}}^2$ available at $\omega = 1$.

**Measured on the second difference matrix**, sweeps to a relative residual of $10^{-10}$, with
the estimated $\omega$ in brackets:

| $n$ | Gauss-Seidel | warmup 30 | warmup 100 | warmup 300 | warmup 1000 | SOR at $\omega^\ast$ |
|---|---|---|---|---|---|---|
| 31 | 2039 | **209** (1.8120) | 308 (1.8226) | 693 (1.8217) | 2012 (1.8215) | 128 ($\omega^\ast$ = 1.8215) |
| 63 | 7728 | 1018 (1.7723) | **545** (1.8921) | 819 (1.9067) | 2188 (1.9065) | 256 (1.9065) |
| 127 | 29192 | 3793 (1.7716) | 2231 (1.8663) | **1494** (1.9321) | 2427 (1.9521) | 512 (1.9521) |

**The warmup length is a real trade-off, and it is the interesting part.** A short warmup
measures a rate still polluted by the transient, which comes out **faster** than the asymptotic
rate, so $\rho_{\text{J}}$ is **underestimated** and $\omega$ comes out too low: 1.7716 against
the true 1.9521 at $n = 127$. A long warmup gets $\omega$ right to four decimals and has already
spent more sweeps than the whole optimal run needs.

**The bias runs in the worst possible direction.** Lesson 23 section 5 measured that the optimum
is asymmetric: guessing $\omega$ too low costs far more than guessing too high. The transient
biases the estimate low, exactly the expensive side, so nudging the estimate upward is the right
correction.

**The best warmup grows with $n$** (30, 100, 300 for $n$ = 31, 63, 127), which is what you would
expect, since the transient lasts a number of sweeps that scales with the problem. At its best
setting the adaptive method costs 1.6 to 3 times the optimal run and still beats Gauss-Seidel by
a factor of 10 to 20 without being told anything.

### 4.1 How ordering changes the Gauss-Seidel rate

Measured on the second difference matrix, sweeps to a relative residual of $10^{-10}$:

| $n$ | natural | reversed | red-black | random |
|---|---|---|---|---|
| 20 | 905 | 905 | 920 | 916 |
| 40 | 3285 | 3285 | 3342 | 3322 |
| 80 | 12145 | 12145 | 12375 | 12333 |

**Reversed is identical to natural, exactly.** That is not luck: reversing the order conjugates
$G$ by the flip permutation $J$, and $JAJ = A$ for the second difference matrix, so the two
iteration matrices are similar and have the same spectrum.

**Red-black and random are within 2 percent** of natural. The second difference matrix is
consistently ordered under all four, so Theorem 23.4 gives $\rho_{\text{GS}} =
\rho_{\text{J}}^2$ for each, and the small differences are transient effects, not rate
differences.

**The general lesson is the opposite of what the numbers suggest.** Ordering barely matters
here because the matrix is highly structured. On a general sparse matrix it matters a great
deal, in the same way and for the same reason that ordering controls fill-in in lesson 21. The
model problem is the wrong place to look for an ordering effect, and that itself is worth
knowing.

### 4.2 The transient before the asymptotic rate

Build $G$ directly with a zero diagonal, so it genuinely is a Jacobi iteration matrix:
superdiagonal 2, subdiagonal 0.1. This is tridiagonal with zero diagonal, whose eigenvalues are
$2\sqrt{ab}\cos(k\pi/(n+1))$ with $a = 2$, $b = 0.1$, so $\rho \approx 0.89$ while
$\|G\|_2 \approx 2.1$.

Measured, starting from a random unit error:

| $n$ | $\rho(G)$ | $\|G\|_2$ | peak error | peak at sweep |
|---|---|---|---|---|
| 8 | 0.8405 | 2.081 | $3.5\times10^{1}$ | 9 |
| 12 | 0.8684 | 2.090 | $1.4\times10^{3}$ | 16 |
| 16 | 0.8792 | 2.094 | $9.8\times10^{4}$ | 27 |
| 24 | 0.8874 | 2.097 | $3.0\times10^{8}$ | 43 |
| 32 | 0.8904 | 2.098 | $2.2\times10^{11}$ | 60 |
| 48 | 0.8926 | 2.099 | $9.1\times10^{18}$ | 97 |

**The transient lasts about $2n$ sweeps and grows like $\|G\|_2^{2n}$.** The reason is that
$G$'s large part is the strictly upper triangle, which is **nilpotent of index $n$**: the
non-normal growth can persist for as long as powers of a nilpotent matrix are nonzero, which is
$n$ steps, and here about $2n$ because both triangles contribute.

**So $\rho(G) < 1$ guarantees eventual convergence and says nothing about when.** For $n = 48$
the error grows by nineteen orders of magnitude first. In floating point that is fatal, because
the initial error's fine structure is destroyed before the asymptotic phase begins.

This is the same phenomenon as lesson 15's warning that $\rho$ is not a norm, and it is exactly
what pseudospectra (lesson 40) exist to quantify. The practical rule: for a strongly non-normal
$G$, the spectral radius is a statement about the limit and the norm is a statement about the
next sweep, and you have to live in both.

### 4.3 Smoothing for Gauss-Seidel and SOR

Measured on the second difference matrix at $n = 63$, applying each iteration matrix to the
$n$ discrete sine modes and recording the shrinkage of each:

| method | $\rho(G)$ | worst shrinkage overall | worst over the upper half |
|---|---|---|---|
| Jacobi | 0.9988 | 0.9988 | **0.9988** |
| damped Jacobi, $\omega = 2/3$ | 0.9995 | 0.9995 | **0.3333** |
| Gauss-Seidel | 0.9976 | 0.9976 | **0.4435** |
| SOR, $\omega = 1.5$ | 0.9928 | 0.9929 | 0.7318 |
| SOR, $\omega^\ast = 1.9065$ | 0.9065 | 1.0898 | **1.0898** |

**Gauss-Seidel is a good smoother**, at 0.4435 over the upper half, close to damped Jacobi's
0.3333. That is why it is the other standard multigrid smoother.

**SOR at the optimal $\omega$ is a bad smoother, and it is worse than useless**: the worst
upper-half shrinkage is **1.0898**, meaning the high frequency modes **grow**. Over-relaxation
overshoots, and overshooting is exactly wrong for the modes the coarse grid cannot fix.

**This decides the question the exercise asks.** SOR is a good solver and a bad smoother. The
two jobs are different: a solver wants the smallest $\rho$, a smoother wants the smallest
shrinkage over the **top half** of the spectrum, and optimising one actively damages the other.
Lesson 28 therefore uses damped Jacobi or Gauss-Seidel and never SOR at $\omega^\ast$.

### 5.1 Consistent ordering and property A

**Property A.** $A$ has property A if the index set $\{1,\dots,n\}$ splits into two disjoint
sets $S_1, S_2$ such that $a_{ij} \ne 0$ with $i \ne j$ implies $i$ and $j$ lie in different
sets. Equivalently, the graph of $A$'s off-diagonal pattern is **bipartite**, and equivalently
there is a permutation making $A$ block form $\begin{pmatrix} D_1 & F \\ E & D_2\end{pmatrix}$
with $D_1, D_2$ diagonal.

**Consistent ordering.** $A$ is consistently ordered if for the splitting $A = D + L + U$, the
eigenvalues of

$$J(\alpha) = \alpha D^{-1}L + \alpha^{-1}D^{-1}U$$

are independent of $\alpha \ne 0$. Every matrix with property A can be permuted to a
consistently ordered form, red-black being one.

**Proof of Theorem 23.4.** Assume $A$ consistently ordered. An eigenvalue $\lambda \ne 0$ of
$G_{\text{SOR}}(\omega)$ satisfies

$$\det\left(\lambda\left(\tfrac{D}{\omega}+L\right) - \left(\tfrac{1-\omega}{\omega}D-U\right)\right)=0.$$

Multiply by $\omega\lambda^{-1/2}$ and factor out $D$:

$$\det\left(D\right)\det\left(\lambda^{1/2}I - \left(\lambda^{1/2}\omega D^{-1}L\cdot\lambda^{-1/2}
+ \lambda^{-1/2}\omega D^{-1}U\cdot\lambda^{1/2}\right)\cdot\frac{1}{\ }\right) \dots$$

More cleanly: the condition rearranges to

$$\frac{(\lambda + \omega - 1)^2}{\lambda\omega^2} = \mu^2$$

where $\mu$ runs over the eigenvalues of $D^{-1}(L+U)$, that is over the Jacobi eigenvalues.
Consistent ordering is exactly what lets $\lambda^{\pm 1/2}$ be absorbed into $L$ and $U$
without changing the spectrum, which is the whole content of the definition.

Setting $\omega = 1$ gives $\lambda = \mu^2$, so

$$\rho_{\text{GS}} = \rho_{\text{J}}^2$$

and Gauss-Seidel is exactly twice as fast. Minimising over $\omega$ gives
$\omega^\ast = 2/(1+\sqrt{1-\rho_{\text{J}}^2})$ and
$\rho(\omega^\ast) = \omega^\ast - 1$.

**A matrix where Gauss-Seidel is slower than Jacobi.** The hypotheses are needed. Collatz's
example is

$$A = \begin{pmatrix} 1 & 2 & -2 \\ 1 & 1 & 1 \\ 2 & 2 & 1 \end{pmatrix}.$$

Measured: $\rho_{\text{J}} = 0$ and $\rho_{\text{GS}} = 2.000000$. **Jacobi converges in three
sweeps exactly and Gauss-Seidel diverges.** The Jacobi matrix here is nilpotent, and computing
$G_{\text{J}}^3$ gives the exact zero matrix, so the spectral radius really is zero and not
merely small. (`spectral_radius` reports $1.1\times10^{-5}$, because the eigenvalues of a
nilpotent matrix can only be found to about $u^{1/3}$, which is lesson 06's multiple root
sensitivity appearing in an eigenvalue problem. Lesson 36 returns to this.)

**And the reverse happens too.** For

$$A = \begin{pmatrix} 2 & -1 & 1 \\ 2 & 2 & 2 \\ -1 & -1 & 2 \end{pmatrix}$$

the measurement gives $\rho_{\text{J}} = 1.118034 > 1$ and $\rho_{\text{GS}} = 0.500000$, so
**Jacobi diverges and Gauss-Seidel converges quickly**. Neither method dominates the other in
general.

Neither matrix is consistently ordered and neither has property A, and Theorem 23.4's conclusion
fails in both directions as badly as it possibly could.

**The takeaway is that "Gauss-Seidel is twice as fast" is a theorem about a class**, not a
general fact, and the class is exactly the one the model problem sits in.

### 5.2 Chebyshev acceleration

**The idea.** A stationary iteration produces $\mathbf{e}^{(k)} = G^k\mathbf{e}^{(0)}$. Take
instead a **combination** $\hat{\mathbf{e}}^{(k)} = p_k(G)\mathbf{e}^{(0)}$ with $p_k$ any
polynomial of degree $k$ satisfying $p_k(1) = 1$ (needed so a zero error stays zero, and so the
iteration is consistent). Minimising $\max_{\lambda \in [-\rho,\rho]}|p_k(\lambda)|$ over such
polynomials is a classical problem whose answer is the shifted Chebyshev polynomial:

$$p_k(\lambda) = \frac{T_k(\lambda/\rho)}{T_k(1/\rho)}.$$

The resulting error bound is $1/T_k(1/\rho)$, which decays like $\sigma^k$ with

$$\sigma = \frac{\rho}{1+\sqrt{1-\rho^2}},$$

exactly as the exercise states.

**Deriving the recurrence.** Chebyshev polynomials satisfy $T_{k+1}(t) = 2tT_k(t) - T_{k-1}(t)$,
which turns into a three-term recurrence on the iterates:

```python
def chebyshev_accelerate(A, b, rho, x0=None, tol=1e-10, max_iter=10000):
    """Chebyshev-accelerated Jacobi. Needs rho = rho(G_Jacobi) supplied in advance.

    Sizes come from A and b. rho must satisfy 0 < rho < 1.
    """
    if not 0.0 < rho < 1.0:
        raise ValueError(f"need 0 < rho < 1, got {rho}")
    A = np.atleast_2d(np.asarray(A, dtype=float))
    b = np.asarray(b, dtype=float).ravel()
    d = np.diag(A).copy()
    n = A.shape[0]
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).ravel().copy()
    scale = float(np.linalg.norm(b)) or 1.0
    x_prev = x.copy()
    mu_prev, mu = 1.0, 1.0 / rho
    for k in range(1, max_iter + 1):
        r = b - A @ x
        if np.linalg.norm(r) <= tol * scale:
            return x, k - 1
        mu_next = 2.0 / rho * mu - mu_prev
        gamma = 2.0 * mu / (rho * mu_next)
        x, x_prev = gamma * (x + r / d) + (1.0 - gamma) * x_prev, x
        mu_prev, mu = mu, mu_next
    return x, max_iter
```

**Measured on the second difference matrix**, sweeps to a relative residual of $10^{-10}$:

| $n$ | Jacobi | Gauss-Seidel | Chebyshev on Jacobi | SOR at $\omega^\ast$ | CG |
|---|---|---|---|---|---|
| 31 | 4146 | 2039 | 271 | 128 | 16 |
| 63 | 15741 | 7728 | 535 | 256 | 32 |
| 127 | 59531 | 29192 | 1057 | 512 | 64 |

And the rates behind them:

| $n$ | $\rho_{\text{J}}$ | $\sigma$ | $\sigma^2$ | $\omega^\ast - 1$ |
|---|---|---|---|---|
| 31 | 0.995185 | 0.906347 | **0.821465** | **0.821465** |
| 63 | 0.998795 | 0.952079 | **0.906455** | **0.906455** |
| 127 | 0.999699 | 0.975753 | **0.952093** | **0.952093** |

**Chebyshev acceleration changes the order, exactly as advertised.** $1 - \sigma$ is $O(h)$
where $1 - \rho_{\text{J}}$ was $O(h^2)$, and the sweep count goes from quadrupling with $n$
(4146, 15741, 59531) to roughly doubling (271, 535, 1057).

**But SOR is exactly twice as fast, and the second table says why.**
$\sigma^2 = \omega^\ast - 1$ to every digit shown, at every $n$. So one SOR sweep at
$\omega^\ast$ achieves what two Chebyshev-accelerated Jacobi sweeps achieve, and the measured
counts confirm it: ratios 2.12, 2.09, 2.06.

That identity is not a coincidence. SOR is in effect Chebyshev acceleration applied to
Gauss-Seidel rather than to Jacobi, and $\rho_{\text{GS}} = \rho_{\text{J}}^2$ is the factor of
two. Feeding a Gauss-Seidel step into the recurrence above **diverges**, because the recurrence
assumes eigenvalues filling a symmetric interval $[-\rho,\rho]$, which Jacobi's do on this
matrix and Gauss-Seidel's, all lying in $[0,\rho_{\text{J}}^2]$, do not. Shifting and rescaling
the interval fixes it and recovers SOR's count.

**And both need what CG does not.** Chebyshev needs the interval
$[\lambda_{\min},\lambda_{\max}]$, and it diverges outright if the interval is wrong. SOR needs
$\rho_{\text{J}}$, and exercise 3.3 measured how expensive a bad estimate is. Conjugate gradient
builds the optimal polynomial **adaptively**, from residuals it has already computed, and needs
no spectral information at all. Measured at $n = 127$: 64 iterations against Chebyshev's 1057, a
factor of 16, with no parameter to get wrong. That is the single most practical reason CG
replaced both.

### 5.3 Why these are the wrong methods

**The argument.** Rewrite Jacobi as

$$\mathbf{x}^{(k+1)} = \mathbf{x}^{(k)} + D^{-1}\mathbf{r}^{(k)},$$

a step of **fixed length 1** in the direction $D^{-1}\mathbf{r}^{(k)}$. SOR is the same with
length $\omega$. In both cases the step length was chosen before any residual was seen.

But once you have chosen a direction $\mathbf{p}$, the best step length along it is a
one-dimensional minimisation you can solve **exactly**. For SPD $A$, minimising
$\phi(\mathbf{x} + \alpha\mathbf{p}) = \tfrac12(\mathbf{x}+\alpha\mathbf{p})^TA
(\mathbf{x}+\alpha\mathbf{p}) - \mathbf{b}^T(\mathbf{x}+\alpha\mathbf{p})$ over $\alpha$ gives

$$\frac{d\phi}{d\alpha} = \mathbf{p}^TA\mathbf{x} + \alpha\mathbf{p}^TA\mathbf{p}
- \mathbf{p}^T\mathbf{b} = 0
\quad\Longrightarrow\quad
\alpha = \frac{\mathbf{p}^T\mathbf{r}}{\mathbf{p}^TA\mathbf{p}}.$$

This costs one extra inner product and one extra matvec, and it can only improve on any fixed
choice, because the fixed choice is one of the $\alpha$ values it minimises over. Taking
$\mathbf{p} = \mathbf{r}$, the steepest descent direction, gives **steepest descent**.

**Measured on the second difference matrix at $n = 63$**, random right-hand side, to a
relative residual of $10^{-10}$. Richardson is $\mathbf{x} \leftarrow \mathbf{x} +
\omega\mathbf{r}$ with $\omega$ fixed in advance:

| method | iterations |
|---|---|
| fixed step $\omega = 0.5$ | 17632 |
| fixed step $\omega = 0.25$ | 34666 |
| fixed step $\omega = 0.1$ | 86681 |
| fixed step $\omega = 0.9$ | **diverges** |
| optimal step every iteration | 17614 |
| conjugate gradient | 63 |

**The optimal step gains 0.1 percent here, and that is the honest answer.** For this matrix
$D = 2I$, so Jacobi's implicit step is $1/2$, and the best possible fixed step is
$2/(\lambda_{\min}+\lambda_{\max}) = 2/4 = 0.5$. **Jacobi is already taking the optimal step**,
so there is nothing for steepest descent to find. Running both from
$\mathbf{b} = A\mathbf{1}$ they agree to the sweep, at 15741 each, because the residual is then
symmetric about the midpoint and $\alpha_k = 0.5$ comes out **exactly** at every step.

**What the optimal step actually buys is not speed but safety.** A step chosen badly costs a
factor of 2 at $\omega = 0.25$, a factor of 5 at $\omega = 0.1$, and diverges at $\omega = 0.9$.
Steepest descent can never do any of that, because it computes the best step rather than
guessing it, and it needs no spectral information to do so. That is the same argument that
separates CG from Chebyshev in exercise 5.2.

So the principle is real and it is **nowhere near enough**.

**Why steepest descent is still not good enough.** Its rate is

$$\|\mathbf{e}_{k+1}\|_A \le \frac{\kappa-1}{\kappa+1}\|\mathbf{e}_k\|_A,$$

which is $1 - 2/\kappa$, so the count is $O(\kappa)$, the same order as Jacobi. The optimal
step buys a constant, not an order.

The reason is visible in the geometry: consecutive steepest descent directions are **exactly
orthogonal** (lesson 24, Proposition 24.2, measured at 90.0000 degrees every time), so the
method zigzags across a narrow valley, and every step partly undoes the previous one. Choosing
each step optimally does not stop later steps from spoiling earlier ones.

**Lesson 24's answer.** Do not minimise along each direction in isolation. Choose directions
that are $A$-conjugate, so that minimising along a new one **never disturbs** the minimisation
already achieved along the old ones. Then $k$ steps solve a $k$-dimensional problem exactly,
and the count drops from $O(\kappa)$ to $O(\sqrt{\kappa})$.

---

## Lesson 24, Conjugate Gradient

### 1.1 Why CG needs positive definiteness and not just symmetry

Three separate places need it.

**The quadratic form must have a minimum.** CG solves $A\mathbf{x} = \mathbf{b}$ by minimising
$\phi(\mathbf{x}) = \tfrac12\mathbf{x}^TA\mathbf{x} - \mathbf{b}^T\mathbf{x}$. If $A$ is
symmetric but indefinite, $\phi$ has a **saddle point**, not a minimum, and there is nothing to
descend to.

**The step length can divide by zero.** $\alpha_k = \mathbf{r}_k^T\mathbf{r}_k /
\mathbf{p}_k^TA\mathbf{p}_k$, and for an indefinite $A$ the denominator
$\mathbf{p}^TA\mathbf{p}$ can be zero or negative for a nonzero $\mathbf{p}$. Zero breaks the
algorithm outright and negative sends it uphill.

**The $A$-inner product must be an inner product.** Conjugacy means
$\langle \mathbf{p}_i, \mathbf{p}_j\rangle_A = \mathbf{p}_i^TA\mathbf{p}_j = 0$, and the whole
theory (Theorem 24.4, independence of conjugate directions, the decomposition into independent
one-dimensional problems) is Gram-Schmidt in that inner product. Positive definiteness is
exactly the axiom $\langle\mathbf{v},\mathbf{v}\rangle > 0$ that makes it an inner product.

**For symmetric indefinite systems the right method is MINRES**, which minimises the residual
in the 2-norm instead of the error in the $A$-norm, and so needs only symmetry. It keeps the
short recurrence, because the Lanczos process needs symmetry alone.

### 1.2 Two matrices with $\kappa = 10^6$: can you predict which is faster?

**No.** $\kappa$ bounds the count but does not determine it, and the gap can be enormous.

Measured in lesson 24 section 6, two matrices with **identical** $\kappa = 10^4$:

| spectrum | iterations |
|---|---|
| geometric, $n$ distinct values in $[1, 10^4]$ | 200 |
| two tight clusters at 1 and $10^4$ | 14 |

The bound $O(\sqrt{\kappa})$ uses only the two endpoints, and it is the best bound expressible
in those two numbers alone. What actually decides the count is how well a low degree polynomial
with $q(0)=1$ can be small on the whole spectrum, which depends on the **shape** of the
spectrum: a few tight clusters are cheap, a dense spread is expensive.

**What you can predict** is the worst case: neither will need more than
$\tfrac12\sqrt{\kappa}\ln(2/\varepsilon)$ iterations, so with $\kappa = 10^6$ and
$\varepsilon = 10^{-8}$, about 9600. Both may do far better.

**This is precisely what preconditioning exploits** (lesson 25): a preconditioner that clusters
the spectrum can beat one that reduces $\kappa$ more.

### 1.3 Why CG needs three vectors and GMRES needs all of them

Because $A$ is symmetric, the Lanczos process reduces $A$ to a **tridiagonal** matrix, so the
orthogonalization of a new Krylov vector against all previous ones requires subtracting only
**two** terms: everything further back is automatically orthogonal.

Concretely, for symmetric $A$, $\mathbf{q}_{j+1}$ needs orthogonalizing against $\mathbf{q}_j$
and $\mathbf{q}_{j-1}$ only, because

$$\mathbf{q}_i^TA\mathbf{q}_j = (A\mathbf{q}_i)^T\mathbf{q}_j = 0 \quad\text{for } i < j-1,$$

since $A\mathbf{q}_i \in \operatorname{span}(\mathbf{q}_1,\dots,\mathbf{q}_{i+1})$, which is
orthogonal to $\mathbf{q}_j$. **Symmetry is the entire reason**, used in the step where
$A$ moves from one side of the inner product to the other.

For nonsymmetric $A$, $H$ is upper Hessenberg rather than tridiagonal, so column $j$ has $j+1$
nonzeros and the new vector must be orthogonalized against **all** previous ones. Storage grows
like $mn$ and work per step like $mn$, which is why GMRES is restarted (lesson 27).

CG therefore keeps $\mathbf{x}$, $\mathbf{r}$ and $\mathbf{p}$, three vectors, at any $n$ and
any iteration count. That is the property that makes $n = 10^6$ possible on a laptop.

### 2.1 Consecutive steepest descent directions are orthogonal

**Claim.** With the exact line search $\alpha_k = \mathbf{r}_k^T\mathbf{r}_k /
\mathbf{r}_k^TA\mathbf{r}_k$, consecutive residuals satisfy
$\mathbf{r}_{k+1}^T\mathbf{r}_k = 0$.

$\mathbf{r}_{k+1} = \mathbf{b} - A(\mathbf{x}_k + \alpha_k\mathbf{r}_k)
= \mathbf{r}_k - \alpha_kA\mathbf{r}_k$, so

$$\mathbf{r}_{k+1}^T\mathbf{r}_k = \mathbf{r}_k^T\mathbf{r}_k
- \alpha_k\mathbf{r}_k^TA\mathbf{r}_k
= \mathbf{r}_k^T\mathbf{r}_k - \frac{\mathbf{r}_k^T\mathbf{r}_k}{\mathbf{r}_k^TA\mathbf{r}_k}
\mathbf{r}_k^TA\mathbf{r}_k = 0.$$

**The geometric reading is the useful one.** Minimising exactly along a line means stopping
where the directional derivative vanishes, and the directional derivative of $\phi$ along
$\mathbf{r}_k$ at the new point is exactly $-\mathbf{r}_{k+1}^T\mathbf{r}_k$. So the exact line
search **forces** the next residual to be orthogonal to the direction just used.

**And that is the flaw, not a feature.** Every step is at a right angle to the last, so on an
elongated ellipse the path zigzags across the valley instead of running down it. Measured in
lesson 24 section 2: 90.0000 degrees, every single step. Only orthogonality to the previous
step is enforced, and nothing prevents step $k+2$ from undoing step $k$.

### 2.2 Proof of Theorem 24.4, including independence

**Claim.** If $\mathbf{p}_0,\dots,\mathbf{p}_{n-1}$ are nonzero and pairwise $A$-conjugate,
they are linearly independent, and the iteration

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \alpha_k\mathbf{p}_k, \qquad
\alpha_k = \frac{\mathbf{p}_k^T\mathbf{r}_k}{\mathbf{p}_k^TA\mathbf{p}_k}$$

reaches the exact solution in at most $n$ steps.

**Independence.** Suppose $\sum_i c_i\mathbf{p}_i = \mathbf{0}$. Multiply by
$\mathbf{p}_j^TA$:

$$0 = \sum_i c_i\mathbf{p}_j^TA\mathbf{p}_i = c_j\,\mathbf{p}_j^TA\mathbf{p}_j,$$

all other terms vanishing by conjugacy. Since $A$ is positive definite and
$\mathbf{p}_j \ne \mathbf{0}$, $\mathbf{p}_j^TA\mathbf{p}_j > 0$, so $c_j = 0$ for every $j$.

**Positive definiteness is essential here**: for an indefinite $A$ a nonzero vector can satisfy
$\mathbf{p}^TA\mathbf{p} = 0$, and then conjugate directions can be dependent.

**Termination.** The $n$ directions form a basis, so write
$\mathbf{x}^\ast - \mathbf{x}_0 = \sum_i \gamma_i\mathbf{p}_i$. Multiply by
$\mathbf{p}_j^TA$:

$$\mathbf{p}_j^TA(\mathbf{x}^\ast - \mathbf{x}_0) = \gamma_j\mathbf{p}_j^TA\mathbf{p}_j
\quad\Longrightarrow\quad
\gamma_j = \frac{\mathbf{p}_j^T(\mathbf{b} - A\mathbf{x}_0)}{\mathbf{p}_j^TA\mathbf{p}_j}
= \frac{\mathbf{p}_j^T\mathbf{r}_0}{\mathbf{p}_j^TA\mathbf{p}_j}.$$

Now note $\mathbf{r}_j = \mathbf{r}_0 - A\sum_{i<j}\alpha_i\mathbf{p}_i$, so
$\mathbf{p}_j^T\mathbf{r}_j = \mathbf{p}_j^T\mathbf{r}_0$ by conjugacy. Hence
$\alpha_j = \gamma_j$ exactly, and after $n$ steps
$\mathbf{x}_n = \mathbf{x}_0 + \sum_j\gamma_j\mathbf{p}_j = \mathbf{x}^\ast$.

**The content of the theorem is the decomposition.** The $n$-dimensional minimisation splits
into $n$ **independent** one-dimensional minimisations, one per direction, and the coefficient
$\gamma_j$ of the exact answer is computable from information available at step $j$. That is
why no step ever needs revisiting, and it is the exact property steepest descent lacks.

### 2.3 CG residuals span the Krylov space

**Claim.** $\operatorname{span}(\mathbf{r}_0,\dots,\mathbf{r}_{k-1})
= \operatorname{span}(\mathbf{p}_0,\dots,\mathbf{p}_{k-1}) = K_k(A,\mathbf{r}_0)$.

By induction. At $k=1$ both sides are $\operatorname{span}(\mathbf{r}_0)$, since
$\mathbf{p}_0 = \mathbf{r}_0$.

Assume it at $k$. Then:

- $\mathbf{r}_k = \mathbf{r}_{k-1} - \alpha_{k-1}A\mathbf{p}_{k-1}$. By hypothesis
  $\mathbf{r}_{k-1} \in K_k$ and $\mathbf{p}_{k-1} \in K_k$, so $A\mathbf{p}_{k-1} \in AK_k
  \subseteq K_{k+1}$. Hence $\mathbf{r}_k \in K_{k+1}$.
- $\mathbf{p}_k = \mathbf{r}_k + \beta_k\mathbf{p}_{k-1} \in K_{k+1}$ likewise.

So both spans are contained in $K_{k+1}$. For the reverse containment, $K_{k+1}$ is spanned by
$K_k$ together with $A^k\mathbf{r}_0$, and $A^k\mathbf{r}_0$ appears in $\mathbf{r}_k$ with the
nonzero coefficient $-\prod_{i<k}\alpha_i$ (nonzero because each $\alpha_i > 0$ while the
method has not terminated). Dimensions therefore match and the spans are equal.

**This is why CG is a Krylov method and not merely an optimisation method.** Combined with
Theorem 24.4 it says $\mathbf{x}_k$ minimises $\|\mathbf{e}\|_A$ over the **whole** affine
space $\mathbf{x}_0 + K_k$, not just along the last direction. That optimality is what makes
exercise 5.1's polynomial characterisation available, and hence the $O(\sqrt{\kappa})$ bound.

### 2.4 Deriving $\alpha$ and $\beta$

**The step length $\alpha_k$.** Impose $\mathbf{r}_{k+1}^T\mathbf{r}_k = 0$ with
$\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_kA\mathbf{p}_k$:

$$0 = \mathbf{r}_k^T\mathbf{r}_k - \alpha_k\mathbf{r}_k^TA\mathbf{p}_k
\quad\Longrightarrow\quad
\alpha_k = \frac{\mathbf{r}_k^T\mathbf{r}_k}{\mathbf{r}_k^TA\mathbf{p}_k}.$$

Simplify the denominator using $\mathbf{p}_k = \mathbf{r}_k + \beta_k\mathbf{p}_{k-1}$ and
$\mathbf{p}_{k-1}^TA\mathbf{p}_k = 0$:

$$\mathbf{p}_k^TA\mathbf{p}_k = \mathbf{r}_k^TA\mathbf{p}_k
+ \beta_k\underbrace{\mathbf{p}_{k-1}^TA\mathbf{p}_k}_{0} = \mathbf{r}_k^TA\mathbf{p}_k,$$

giving the standard form

$$\alpha_k = \frac{\mathbf{r}_k^T\mathbf{r}_k}{\mathbf{p}_k^TA\mathbf{p}_k}.$$

**The direction coefficient $\beta_{k+1}$.** Impose
$\mathbf{p}_{k+1}^TA\mathbf{p}_k = 0$ with
$\mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_{k+1}\mathbf{p}_k$:

$$0 = \mathbf{r}_{k+1}^TA\mathbf{p}_k + \beta_{k+1}\mathbf{p}_k^TA\mathbf{p}_k
\quad\Longrightarrow\quad
\beta_{k+1} = -\frac{\mathbf{r}_{k+1}^TA\mathbf{p}_k}{\mathbf{p}_k^TA\mathbf{p}_k}.$$

Now use $A\mathbf{p}_k = (\mathbf{r}_k - \mathbf{r}_{k+1})/\alpha_k$:

$$\beta_{k+1} = -\frac{\mathbf{r}_{k+1}^T(\mathbf{r}_k - \mathbf{r}_{k+1})}
{\alpha_k\mathbf{p}_k^TA\mathbf{p}_k}
= \frac{\mathbf{r}_{k+1}^T\mathbf{r}_{k+1}}{\mathbf{r}_k^T\mathbf{r}_k},$$

using $\mathbf{r}_{k+1}^T\mathbf{r}_k = 0$ and
$\alpha_k\mathbf{p}_k^TA\mathbf{p}_k = \mathbf{r}_k^T\mathbf{r}_k$.

**Notice what has happened.** Two local conditions, one orthogonality and one conjugacy against
only the immediately previous vector, produce coefficients that make the new direction
conjugate to **all** previous directions and the new residual orthogonal to **all** previous
residuals. That is the miracle of CG, and it rests entirely on the symmetry that gives Lanczos
its three-term recurrence.

The final forms cost one matvec and two inner products per step, and reference nothing older
than one step back.

### 2.5 Proof of Theorem 24.6 with Chebyshev polynomials

By exercise 5.1, $\|\mathbf{e}_k\|_A = \min_{q}\|q(A)\mathbf{e}_0\|_A$ over polynomials of
degree $\le k$ with $q(0) = 1$. Diagonalize $A = Q\Lambda Q^T$ and expand
$\mathbf{e}_0 = \sum_i c_i\mathbf{q}_i$:

$$\|q(A)\mathbf{e}_0\|_A^2 = \sum_i \lambda_i c_i^2 q(\lambda_i)^2
\le \Big(\max_i |q(\lambda_i)|\Big)^2\|\mathbf{e}_0\|_A^2,$$

so

$$\frac{\|\mathbf{e}_k\|_A}{\|\mathbf{e}_0\|_A}
\le \min_{q}\max_{\lambda \in [\lambda_{\min},\lambda_{\max}]}|q(\lambda)|.$$

Since CG's polynomial is the minimiser, **any** admissible $q$ gives an upper bound. Choose the
shifted and scaled Chebyshev polynomial:

$$q_k(\lambda) = \frac{T_k\!\left(\dfrac{\lambda_{\max}+\lambda_{\min}-2\lambda}
{\lambda_{\max}-\lambda_{\min}}\right)}
{T_k\!\left(\dfrac{\lambda_{\max}+\lambda_{\min}}{\lambda_{\max}-\lambda_{\min}}\right)}.$$

The argument maps $[\lambda_{\min},\lambda_{\max}]$ onto $[-1,1]$, where $|T_k| \le 1$, and
$q_k(0) = 1$ by construction. So the bound is $1/T_k(z)$ with
$z = (\kappa+1)/(\kappa-1)$.

For $z > 1$, $T_k(z) = \cosh(k\operatorname{arccosh} z) = \tfrac12(w^k + w^{-k})$ where
$w = z + \sqrt{z^2-1}$. A short calculation gives
$w = (\sqrt{\kappa}+1)/(\sqrt{\kappa}-1)$, so $T_k(z) \ge \tfrac12 w^k$ and

$$\frac{\|\mathbf{e}_k\|_A}{\|\mathbf{e}_0\|_A}
\le 2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k.$$

**Reading the rate.** $(\sqrt{\kappa}-1)/(\sqrt{\kappa}+1) \approx 1 - 2/\sqrt{\kappa}$, so
reaching $\varepsilon$ needs about $\tfrac12\sqrt{\kappa}\ln(2/\varepsilon)$ iterations. That
is $O(\sqrt{\kappa})$, against steepest descent's $1-2/\kappa$ and $O(\kappa)$.

**Where $\sqrt{\kappa}$ comes from** is the Chebyshev polynomial's exponential growth outside
$[-1,1]$: no other polynomial grows as fast at a point outside an interval while staying
bounded on it. The square root is the Joukowski map $z \mapsto z+\sqrt{z^2-1}$ appearing in
$T_k$'s closed form, and nothing about the algorithm.

**The bound is never violated and is very loose**, measured in lesson 24 section 5 at
$10^{-14}$ of the bound near the end of a run, because it discards every eigenvalue except the
two extremes.

### 3.1 Matrix-free CG at $n = 10^6$

CG touches $A$ only through $A\mathbf{p}$, so an operator suffices:

```python
def poisson_2d_operator(m):
    """The five point Laplacian on an m by m grid, as a matvec. No matrix stored.

    m is a free parameter and n = m*m follows from it, so the same code covers every size.
    """
    def apply(v):
        g = v.reshape(m, m)
        out = 4.0 * g
        out[:-1, :] -= g[1:, :]
        out[1:, :]  -= g[:-1, :]
        out[:, :-1] -= g[:, 1:]
        out[:, 1:]  -= g[:, :-1]
        return out.ravel()
    return LinearOperator(apply, shape=(m * m, m * m))
```

**Measured at $m = 1000$, so $n = 10^6$**, to a relative residual of $10^{-8}$:

| quantity | value |
|---|---|
| iterations | 1853 |
| wall clock | 32.0 s |
| CG storage, four vectors of $10^6$ doubles | **32 MB** |
| matrix storage if CSR with 5 nonzeros per row | 64 MB |
| matrix storage if dense | $8\times10^{12}$ bytes, 8 TB |
| dense factorization, $\tfrac13n^3$ flops | $3.3\times10^{17}$ |

**The matrix-free version uses less memory than even the sparse matrix does**, by a factor of
two, because the stencil lives in the code rather than in an array. It is also faster per
iteration, since it reads $n$ doubles instead of $5n$ doubles plus $5n$ indices.

**The point is the ratio.** Dense storage is 250000 times larger than what CG needs, and the
dense factorization at $3.3\times10^{17}$ flops would take about 38 days on a machine doing
$10^{11}$ flop/s, against the measured 32 seconds.

### 3.2 The Polak-Ribiere variant

$$\beta_{k+1}^{\text{PR}} = \frac{\mathbf{r}_{k+1}^T(\mathbf{r}_{k+1}-\mathbf{r}_k)}
{\mathbf{r}_k^T\mathbf{r}_k}, \qquad
\beta_{k+1}^{\text{FR}} = \frac{\mathbf{r}_{k+1}^T\mathbf{r}_{k+1}}{\mathbf{r}_k^T\mathbf{r}_k}.$$

**Identical in exact arithmetic**, because $\mathbf{r}_{k+1}^T\mathbf{r}_k = 0$ exactly, so the
extra term vanishes. The two differ by precisely the quantity that measures how much
orthogonality has been lost.

```python
def cg_polak_ribiere(A, b, tol=1e-10, max_iter=None):
    """CG with the Polak-Ribiere beta. Sizes come from b; max_iter defaults to n."""
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    max_iter = 2 * n if max_iter is None else max_iter
    x = np.zeros(n)
    r = b - A @ x
    p = r.copy()
    rs = r @ r
    scale = float(np.linalg.norm(b)) or 1.0
    for k in range(1, max_iter + 1):
        Ap = A @ p
        alpha = rs / (p @ Ap)
        x += alpha * p
        r_new = r - alpha * Ap
        if np.linalg.norm(r_new) <= tol * scale:
            return x, k
        beta = max(0.0, (r_new @ (r_new - r)) / rs)     # the restart is part of the method
        p = r_new + beta * p
        r, rs = r_new, r_new @ r_new
    return x, max_iter
```

**Measured on SPD matrices with a geometric spectrum, $n = 60$**, iterations to $10^{-10}$
with a cap of $4n = 240$:

| $\kappa$ | Fletcher-Reeves | Polak-Ribiere | PR without the clip |
|---|---|---|---|
| $10^{2}$ | 72 | 72 | 72 |
| $10^{3}$ | 124 | 126 | 126 |
| $10^{4}$ | 205 | 207 | 207 |
| $10^{6}$ | did not converge | did not converge | did not converge |
| $10^{8}$ | did not converge | did not converge | did not converge |

**The measurement contradicts the usual claim, and that is the result.** The two variants are
the same to within two iterations everywhere, and Polak-Ribiere is very slightly **worse**, not
better. The clip never fires:

| $\kappa$ | steps taken | times $\beta$ went negative | largest $\lvert\beta_{\text{PR}}-\beta_{\text{FR}}\rvert / \beta_{\text{FR}}$ |
|---|---|---|---|
| $10^{2}$ | 71 | **0** | $1.7\times10^{-14}$ |
| $10^{4}$ | 205 | **0** | $3.0\times10^{-12}$ |
| $10^{6}$ | 240 | **0** | $1.5\times10^{-11}$ |
| $10^{8}$ | 240 | **0** | $1.6\times10^{-9}$ |

**Why the theory and the measurement disagree.** The extra term is
$-\mathbf{r}_{k+1}^T\mathbf{r}_k / \mathbf{r}_k^T\mathbf{r}_k$, and although CG's residuals lose
orthogonality catastrophically over the whole run (exercise 4.3 measures 0.99 at
$\kappa = 10^8$), **consecutive** residuals stay orthogonal to near machine precision, because
$\alpha_k$ is computed precisely to enforce it at each step. The loss is against **older**
residuals, which $\beta_{\text{PR}}$ does not look at. So the correction term stays at the
$10^{-9}$ level and changes nothing.

**Where Polak-Ribiere genuinely wins is nonlinear CG**, minimising a non-quadratic $f$. There
the "matrix" is the Hessian, which **changes between steps**, so
$\mathbf{r}_{k+1}^T\mathbf{r}_k$ is not small for any algorithmic reason, $\beta$ does go
negative, and the automatic restart is real. On a genuine linear SPD system it is a solution to
a problem that does not arise.

**The general lesson is worth more than the exercise.** A robustness argument that is correct in
principle can be empty in practice, and only a measurement distinguishes the two. Reporting
"Polak-Ribiere is more robust" without checking whether the clip ever fires would have been
repeating a claim rather than testing one.

### 3.3 Ritz values from a CG run

The Lanczos tridiagonal is recoverable from CG's scalars:

$$T_k = \begin{pmatrix}
\frac{1}{\alpha_0} & \frac{\sqrt{\beta_1}}{\alpha_0} & & \\
\frac{\sqrt{\beta_1}}{\alpha_0} & \frac{1}{\alpha_1}+\frac{\beta_1}{\alpha_0} & \ddots & \\
& \ddots & \ddots & \frac{\sqrt{\beta_{k-1}}}{\alpha_{k-2}} \\
& & \frac{\sqrt{\beta_{k-1}}}{\alpha_{k-2}} & \frac{1}{\alpha_{k-1}}+\frac{\beta_{k-1}}{\alpha_{k-2}}
\end{pmatrix}.$$

```python
def ritz_from_cg(alphas, betas):
    """Eigenvalue estimates of A from the alpha and beta a CG run already computed.

    Length is taken from the inputs, so this works after any number of iterations.
    """
    a = np.asarray(alphas, dtype=float)
    b = np.asarray(betas, dtype=float)
    k = a.size
    diag = 1.0 / a
    diag[1:] += b[:k-1] / a[:k-1]
    off = np.sqrt(b[:k-1]) / a[:k-1]
    return np.linalg.eigvalsh(np.diag(diag) + np.diag(off, 1) + np.diag(off, -1))
```

**Measured on an SPD matrix with $n = 60$ and spectrum uniform in $[1, 100]$:**

| CG iteration | smallest Ritz | largest Ritz | relative error in $\lambda_{\min}$ | in $\lambda_{\max}$ |
|---|---|---|---|---|
| 5 | 7.1105 | 94.7164 | $6.1\times10^{0}$ | $5.3\times10^{-2}$ |
| 10 | 2.2447 | 99.7680 | $1.2\times10^{0}$ | $2.3\times10^{-3}$ |
| 20 | 1.0011 | 99.9978 | $1.1\times10^{-3}$ | $2.2\times10^{-5}$ |
| 40 | 1.0000 | 100.0000 | $1.6\times10^{-12}$ | $1.7\times10^{-14}$ |

**The largest converges far faster than the smallest**, by two orders of magnitude at every
iteration count. That is the Kaniel-Paige behaviour of exercise 26.2.5: convergence speed
depends on the **gap** to the neighbouring eigenvalue relative to the spread of the spectrum,
and the top of a uniform spectrum is much better separated in that relative sense than the
bottom.

**This is free information.** The $\alpha$ and $\beta$ are already computed, so eigenvalue
estimates cost nothing beyond a small tridiagonal eigensolve. Practical uses: estimating
$\kappa$ to check a stopping rule, and deciding whether deflation would pay (exercise 25.5.1).

### 4.1 The exponent in $\sqrt{\kappa}$

The measurement has to avoid two traps. Finite termination caps the count at $n$, so $n$ must
be far larger than the count. And CG depends only on the spectrum, so a **diagonal** operator
is exactly as good as a dense SPD matrix and costs $O(n)$ per matvec instead of $O(n^2)$.

```python
def diag_op(lams):
    """CG on diag(lams) runs identically to CG on Q diag(lams) Q^T with a rotated b."""
    lams = np.asarray(lams, dtype=float)
    return LinearOperator(lambda v: lams * v, shape=(lams.size, lams.size))
```

**Measured at $n = 20000$**, spectrum uniform in $[1,\kappa]$, tolerance $10^{-8}$, the cap
never reached:

| $\kappa$ | iterations | iterations / $\sqrt{\kappa}$ |
|---|---|---|
| $10^{1}$ | 29 | 9.171 |
| $10^{1.5}$ | 51 | 9.069 |
| $10^{2}$ | 91 | 9.100 |
| $10^{2.5}$ | 160 | 8.998 |
| $10^{3}$ | 274 | 8.665 |
| $10^{3.5}$ | 468 | 8.322 |
| $10^{4}$ | 661 | 6.610 |
| $10^{5}$ | 867 | 2.742 |
| $10^{6}$ | 936 | 0.936 |

**Fitted exponent over $\kappa \in [10, 3\times10^3]$: 0.4843**, against the theoretical 0.5.

**And the constant matches too.** Theory says
$k \approx \tfrac12\sqrt{\kappa}\ln(2/\varepsilon)$, which for $\varepsilon = 10^{-8}$ gives
the constant $\tfrac12\ln(2\times10^8) = 9.557$. Measured: **9.17**. So the bound is tight to
within 4 percent on this family, which is unusual and worth noticing, because on the clustered
spectra of section 6 it was off by orders of magnitude.

**Why the ratio collapses beyond $\kappa \approx 10^4$.** With $n = 20000$ points spread over
$[1,\kappa]$, the spacing near the bottom is $\kappa/n$. Once that exceeds about 1, the smallest
eigenvalues are **isolated** rather than part of a continuum, and CG deflates them individually
instead of paying the Chebyshev price. The bound assumes a filled interval; a sparse one is
easier.

**So the honest answer to "is the exponent $1/2$" is: yes, for a spectrum dense in its
interval, and the measurement confirms both the exponent and the constant.** Outside that
regime CG beats its own bound, which it is allowed to do.

### 4.2 What one outlier costs

Two experiments, both at $n = 60$ with a bulk spectrum uniform in $[1,10]$.

**A tight cluster of outliers, all at $10^4$:**

| outliers | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| iterations | 32 | 40 | 39 | 39 | 39 | 39 | 38 |

**The count does not depend on how many there are.** One polynomial root at $10^4$ annihilates
all of them at once, so a cluster of any size costs what a single eigenvalue costs.

**Distinct, well separated outliers**, each needing its own root. Incremental cost depends on
how far out they sit:

| outliers | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| at $20\times$ the bulk | 32 | 34 | 36 | 38 | 40 | 42 | 46 |
| at $10^2$ | 32 | 35 | 39 | 43 | 46 | 51 | 53 |
| at $10^3$ | 32 | 38 | 42 | 49 | 55 | 60 | 66 |
| at $10^4$ | 32 | 40 | 46 | 55 | 63 | 71 | 78 |
| at $10^6$ | 32 | 44 | 54 | 66 | 78 | 91 | 102 |

**Incremental cost per outlier: 2 steps at $20\times$, 3 at $10^2$, 6 at $10^3$, 8 at $10^4$,
12 at $10^6$.**

In **exact** arithmetic the answer would be 1 per outlier, since one extra polynomial degree
places one extra root. The excess is roundoff: once a root has been placed, loss of
orthogonality lets the annihilated eigenvalue **come back**, and CG has to place the root again.
The further out the eigenvalue, the larger $\kappa$, the faster orthogonality goes, and the more
often it has to be re-annihilated. That is Greenbaum's phenomenon (exercise 5.2) showing up as a
cost.

**The basis for deflation** (exercise 25.5.1) is the first table. If a handful of eigenvalues
dominate $\kappa$, projecting them out of the Krylov space removes their cost entirely, and
because a cluster costs the same as a point, deflating a **cluster** is as cheap as deflating
one eigenvalue. That is why deflation with approximate eigenvectors works.

### 4.3 Loss of orthogonality against $\kappa$

Measured at $n = 60$, geometric spectrum in $[1,\kappa]$, tracking
$\max_{i \ne j}|\hat{\mathbf{r}}_i^T\hat{\mathbf{r}}_j|$ over the normalized residuals produced
so far, stopping at a relative residual of $10^{-10}$ or $4n$ iterations:

| $\kappa$ | iterations | worst orthogonality at half way | at the end | final relative error |
|---|---|---|---|---|
| $10^{1}$ | 32 | $6.2\times10^{-15}$ | $7.4\times10^{-8}$ | $1.2\times10^{-10}$ |
| $10^{2}$ | 66 | $3.9\times10^{-1}$ | $7.6\times10^{-1}$ | $9.0\times10^{-10}$ |
| $10^{3}$ | 117 | $9.0\times10^{-1}$ | $9.1\times10^{-1}$ | $6.4\times10^{-9}$ |
| $10^{4}$ | 195 | $9.4\times10^{-1}$ | $9.6\times10^{-1}$ | $4.0\times10^{-9}$ |
| $10^{6}$ | did not reach $10^{-10}$ | $9.8\times10^{-1}$ | $9.9\times10^{-1}$ | $1.0\times10^{-2}$ |
| $10^{8}$ | did not reach $10^{-10}$ | $9.9\times10^{-1}$ | $9.9\times10^{-1}$ | $2.5\times10^{-1}$ |

**Two separate things are happening, and separating them is the answer to the exercise.**

**Orthogonality is lost long before convergence stalls.** At $\kappa = 10^2$ the residuals are
already 40 percent non-orthogonal half way through, and yet CG converges to $9\times10^{-10}$
without difficulty. At $\kappa = 10^3$ and $10^4$ orthogonality is essentially **gone**, above
0.9, and CG still converges to $10^{-9}$. So loss of orthogonality by itself is not fatal.

**What is fatal is when it happens relative to the count $\sqrt{\kappa}$ demands.** Loss of
orthogonality costs extra iterations, because eigenvalues already annihilated come back
(exercise 4.2). CG survives as long as it can afford them. At $\kappa \ge 10^6$ the required
count exceeds what the corrupted recurrence can deliver, and convergence stops with a relative
error of $10^{-2}$.

**The relationship is therefore not "orthogonality lost, so stall".** It is: orthogonality loss
raises the iteration count, and the method fails when the raised count exceeds the budget. That
is exactly Greenbaum's result (exercise 5.2), that finite precision CG behaves like exact CG on
a larger matrix with clusters in place of eigenvalues, and clusters cost more steps than points
do.

### 5.1 CG as a polynomial approximation problem

**Claim.** $\|\mathbf{e}_k\|_A = \min\{\|q(A)\mathbf{e}_0\|_A : \deg q \le k,\ q(0) = 1\}$.

**Setup.** By exercise 2.3, $\mathbf{x}_k \in \mathbf{x}_0 + K_k(A,\mathbf{r}_0)$, so
$\mathbf{x}_k - \mathbf{x}_0 = p(A)\mathbf{r}_0$ for some polynomial $p$ of degree $< k$. Since
$\mathbf{r}_0 = A\mathbf{e}_0$ with $\mathbf{e}_0 = \mathbf{x}^\ast - \mathbf{x}_0$,

$$\mathbf{e}_k = \mathbf{e}_0 - p(A)A\mathbf{e}_0 = q(A)\mathbf{e}_0,
\qquad q(t) = 1 - t\,p(t).$$

Every such $q$ has degree $\le k$ and $q(0) = 1$, and conversely every polynomial with those
two properties factors as $1 - tp(t)$. So the set of reachable errors is exactly
$\{q(A)\mathbf{e}_0\}$ over that class.

**Optimality.** By Theorem 24.4 and exercise 2.3, $\mathbf{x}_k$ minimises $\phi$, equivalently
$\|\mathbf{e}\|_A$, over the entire affine space $\mathbf{x}_0 + K_k$. Since that space is
exactly the reachable set, CG attains the minimum. Hence the equality.

**Why this makes CG optimal among Krylov methods for SPD systems.** Any method whose $k$-th
iterate lies in $\mathbf{x}_0 + K_k$ produces an error of the form $q(A)\mathbf{e}_0$ with the
same constraints, so its $A$-norm error is at least CG's. And by exercise 26.5.2, no method
using $k$ matrix-vector products can leave $K_k$ at all. So CG is optimal in the $A$-norm over
everything reachable with the information available.

**Two consequences worth stating.**

First, **the bound in Theorem 24.6 is obtained by choosing any convenient $q$**, and the
Chebyshev choice is convenient rather than special. Better choices exist when more is known
about the spectrum, which is exactly how the clustering results of section 6 are proved: put a
root at each cluster.

Second, **the $A$-norm is not a choice**, it is what CG minimises. In the 2-norm CG is **not**
optimal, and its residual norm can increase from one step to the next. MINRES minimises the
2-norm of the residual instead and has a non-increasing residual, which is why it is preferred
when the stopping rule is residual based.

### 5.2 Greenbaum's analysis

**The result.** Greenbaum (1989) showed that finite precision CG applied to $A$ produces the
same iterates, to within a small multiple of the unit roundoff, as **exact** CG applied to a
larger matrix $\hat{A}$ whose eigenvalues lie in tight clusters of width $O(u\|A\|)$ around the
eigenvalues of $A$. The dimension of $\hat{A}$ is not fixed in advance and grows with the number
of iterations.

**Why this explains the observations.** A cluster of $m$ eigenvalues in exact arithmetic costs
more Krylov steps than a single eigenvalue does when the cluster has nonzero width: one root
kills a point exactly, but only approximately kills a spread. So replacing each eigenvalue by a
cluster raises the iteration count, which is precisely what section 8 measures. It also explains
why the **final attainable accuracy** is limited: exact CG on $\hat{A}$ solves a slightly
different problem.

**An experiment that supports it.** The prediction is quantitative: run exact CG (in high
precision) on an artificially clustered spectrum, and compare with finite precision CG on the
unclustered one.

```python
from fractions import Fraction  # or mpmath for a practical high precision run

def clustered(lams, width, per_cluster, rng):
    """Replace each eigenvalue by `per_cluster` values within `width` of it."""
    lams = np.asarray(lams, dtype=float)
    spread = width * (2 * rng.random((lams.size, per_cluster)) - 1)
    return (lams[:, None] * (1.0 + spread)).ravel()
```

Run the same CG twice: once in double precision, and once in 60 decimal digits, where
roundoff is negligible and the run is effectively exact. Then run the exact one again on a
spectrum where each eigenvalue has been replaced by a cluster of three.

**Measured**, $n = 40$, spectrum geometric in $[1,10^4]$, tolerance $10^{-8}$:

| run | iterations |
|---|---|
| **double precision CG on $A$** | **97** |
| 60-digit CG on $A$ | 45 |
| 60-digit CG on clustered $\hat{A}$, width $10^{-14}$ | 84 |
| 60-digit CG on clustered $\hat{A}$, width $10^{-12}$ | 92 |
| **60-digit CG on clustered $\hat{A}$, width $10^{-10}$** | **96** |
| 60-digit CG on clustered $\hat{A}$, width $10^{-8}$ | 105 |

**Exact CG on $A$ needs 45 iterations; double precision CG needs 97, more than twice as many.**
That gap is entirely roundoff, and it is what the theory has to explain.

**Exact CG on the clustered $\hat{A}$ reproduces it: 96 against 97, at a cluster width of
$10^{-10}$.** So the finite precision run really does behave like an exact run on a clustered
matrix, and the clustering costs precisely the iterations roundoff was costing.

**The count responds monotonically to the width** (84, 92, 96, 105 for widths $10^{-14}$ through
$10^{-8}$), which is what makes this evidence rather than a coincidence. A single matching
number would prove nothing, since the width is a free parameter that could always be fitted;
the behaviour of the whole curve is the test.

**The width is larger than $u$, and that is expected.** Roundoff enters through the matvec and
the inner products, so the perturbation is of size $u\|A\|$ relative to the smallest eigenvalue,
which amplifies the effective cluster width by roughly $\kappa$. Here $u\kappa \approx 10^{-12}$
and the fitted width is $10^{-10}$, the same order.

**The design point.** Vary the width and check the response. Do not tune a free parameter to one
number and call it agreement.

### 5.3 CG against Cholesky

The four variables and what each does.

**$n$ alone decides nothing.** Cholesky on a dense SPD matrix costs $\tfrac13n^3$ flops and
$n^2/2$ storage. CG costs $k$ matvecs, so $2kn^2$ flops dense or $2k\cdot\text{nnz}$ sparse.
Dense against dense, CG wins only when $k < n/6$, which needs a well conditioned or clustered
spectrum.

**Sparsity is the real variable.** For a banded matrix of bandwidth $p$, Cholesky costs
$O(np^2)$ and is often unbeatable. For the 2D model problem with nested dissection ordering,
Cholesky costs $O(n^{3/2})$ with $O(n\log n)$ fill. CG costs $O(n)$ per iteration times
$O(\sqrt{\kappa}) = O(n^{1/2})$ iterations, so $O(n^{3/2})$ as well. **They tie**, and the
winner is decided by constants and memory. In 3D, Cholesky is $O(n^2)$ with $O(n^{4/3})$ fill
while CG stays $O(n^{4/3})$, and CG wins decisively.

**$\kappa$ only affects CG.** Cholesky's cost is fixed by the sparsity pattern and does not
depend on $\kappa$ at all; only its **accuracy** does, and that is $\kappa u$ regardless. CG's
cost is $O(\sqrt{\kappa})$, and CG can fail to converge to the requested tolerance at all when
$\kappa \gtrsim 10^{6}$ (exercise 4.3, measured relative error $10^{-2}$ at $\kappa = 10^6$).
**So high $\kappa$ favours the direct method**, which is the opposite of the folk rule.

**Multiple right-hand sides strongly favour Cholesky.** The factorization is computed once and
each extra solve costs $O(n^2)$ dense or $O(\text{nnz}(L))$ sparse. CG must run again from
scratch for every right-hand side. With $s$ right-hand sides the comparison becomes
$\tfrac13n^3 + sn^2$ against $2skn^2$, so Cholesky wins for $s \gtrsim n/(6k)$ even when CG wins
for a single solve. Block CG and Krylov subspace recycling narrow this but do not close it.

**Measured**, 2D Laplacian on an $m \times m$ grid, single right-hand side, tolerance
$10^{-10}$, everything stored **densely** so the comparison is like for like:

| $m$ | $n$ | dense Cholesky | CG | CG with IC(0) |
|---|---|---|---|---|
| 16 | 256 | **0.78 ms** | 59 iters, 2.6 ms | 23 iters, 24.8 ms |
| 36 | 1296 | **15.2 ms** | 129 iters, 19.2 ms | 45 iters, 295 ms |
| 64 | 4096 | **289 ms** | 226 iters, 1285 ms | 76 iters, 6694 ms |

**Cholesky wins at every size, and the flop counts say it should not.** At $n = 4096$, CG did
$226 \times 2n^2 = 7.6\times10^9$ flops and Cholesky did $n^3/3 = 2.3\times10^{10}$, three times
more, and Cholesky was four times faster. The achieved rates explain it: **79 GFlop/s for
Cholesky against 5.9 GFlop/s for CG**, a factor of 13.

That factor is lesson 08's roofline, not anything about linear algebra. Cholesky is one blocked
BLAS-3 call with arithmetic intensity $O(n)$, running near peak. CG is 226 separate BLAS-2
matrix-vector products with intensity 0.25, running at memory bandwidth. **Flop counts predict
the wrong winner because they price the wrong resource.**

The IC(0) column is slower still because the implementation here is the dense $O(n^3)$ one, and
it adds two dense triangular solves per iteration. Exercise 25.3.1 is exactly the fix, and
without it a preconditioner that cuts iterations by a factor of 3 costs a factor of 10 in time.

**None of this survives going sparse.** With CSR storage the CG matvec drops from $2n^2$ to
$10n$ flops, a factor of 800 at $n = 4096$, while sparse Cholesky still pays for fill. The table
above is the dense regime, and it exists to show that the regime is what decides, not $n$.

**Why "iterative for large, direct for small" is too crude.** It is a statement about $n$ alone,
and $n$ is the least informative of the four variables. The rule fails in at least four ways:

- A large, very ill conditioned system where CG cannot reach the tolerance at all.
- A large banded system where Cholesky is $O(np^2)$ and linear in $n$.
- A small dense system solved for thousands of right-hand sides, where factoring once wins.
- A large system with a good preconditioner, where CG wins by far more than the rule suggests,
  so the rule is right for the wrong reason.

**The better rule.** Use a direct method when the factorization fits in memory and you can
afford the fill; use an iterative method when it does not, or when a preconditioner exploiting
the problem's origin (exercise 25.5.3) makes the iteration count small and nearly independent of
$n$. Memory, not flops, is usually what actually decides.

---

## Lesson 25, Preconditioning

### 1.1 Why a CG preconditioner must be symmetric positive definite

Preconditioned CG runs the ordinary algorithm in the $M^{-1}$-inner product, or equivalently on
the split system $M_1^{-1}AM_1^{-T}\mathbf{y} = M_1^{-1}\mathbf{b}$ with $M = M_1M_1^T$. Three
things break if $M$ is not SPD.

**The split matrix must be SPD.** $M_1^{-1}AM_1^{-T}$ is symmetric for any invertible $M_1$, and
it is positive definite exactly when $A$ is. But $M_1$ exists as a real factor only if $M$ is
positive definite, so without that there is no split form at all.

**The preconditioned inner product must be an inner product.** PCG replaces
$\mathbf{r}^T\mathbf{r}$ with $\mathbf{r}^TM^{-1}\mathbf{r} = \mathbf{z}^T\mathbf{r}$ in the
coefficients

$$\alpha_k = \frac{\mathbf{r}_k^T\mathbf{z}_k}{\mathbf{p}_k^TA\mathbf{p}_k}, \qquad
\beta_{k+1} = \frac{\mathbf{r}_{k+1}^T\mathbf{z}_{k+1}}{\mathbf{r}_k^T\mathbf{z}_k}.$$

If $M^{-1}$ is not positive definite, $\mathbf{r}^TM^{-1}\mathbf{r}$ can be zero or negative for
a nonzero residual, so $\alpha$ can be negative and $\beta$ can divide by zero. Exercise 24.1.1
is the same objection one level down.

**Symmetry is what preserves the short recurrence.** If $M^{-1}A$ is not self-adjoint in some
inner product, the Lanczos three-term recurrence does not apply and there is no reason for the
new direction to be conjugate to all the old ones. You would need GMRES (lesson 27) and full
storage.

**In practice this rules out things people reach for.** A single Gauss-Seidel sweep is not a
symmetric preconditioner, which is exactly why SSOR does a forward sweep followed by a backward
one: the pair is symmetric where either alone is not. An incomplete LU is not symmetric either,
which is why the SPD case uses incomplete **Cholesky**.

### 1.2 Halved iterations at triple the cost per iteration

**No, not on those numbers alone.** Total work is iterations times cost per iteration, so
halving one and tripling the other gives $0.5 \times 3 = 1.5$, a 50 percent **loss**.

**But three things can change the answer.**

**Memory traffic rather than flops.** If the preconditioner reuses data already in cache while
the matvec streams from memory, the "triple cost" measured in flops may be far less than triple
in time. Lesson 08's roofline is the tool.

**Setup cost amortised over many right-hand sides.** A preconditioner built once and used for
hundreds of solves changes the arithmetic completely.

**Robustness.** If the unpreconditioned run does not converge at all, as measured in exercise
24.4.3 for $\kappa \ge 10^6$, then a factor of 1.5 in a run that finishes beats any factor in a
run that does not.

**The measured version of this exercise is 4.2**, where SSOR is found to halve the count at
every $\kappa$ tested while roughly doubling the cost per iteration, leaving it almost exactly
break even.

### 1.3 Why Jacobi does nothing to the second difference matrix

Jacobi preconditioning uses $M = D = \operatorname{diag}(A)$. For the second difference matrix
every diagonal entry is 2, so

$$M = 2I, \qquad M^{-1}A = \tfrac12 A.$$

Scaling a matrix by a constant leaves its condition number unchanged and scales every eigenvalue
equally, so the spectrum has exactly the same **shape**. CG is invariant under scaling of $A$
(both $\alpha$ and $\beta$ are ratios that absorb the factor), so the iterates are literally
identical.

**Measured on the 2D Laplacian**, where the diagonal is uniformly 4:

| $m$ | $n$ | no preconditioner | Jacobi |
|---|---|---|---|
| 8 | 64 | 29 | 29 |
| 16 | 256 | 58 | 58 |
| 36 | 1296 | 126 | 126 |

Identical at every size, not merely close.

**The general statement is that Jacobi helps exactly when the diagonal varies.** It is a
**scaling** preconditioner, so it fixes problems caused by badly scaled rows, which is common in
matrices assembled from physical quantities in mixed units, and does nothing at all for a
problem whose difficulty is in the off-diagonal structure. The model problem's difficulty is
entirely structural.

### 2.1 $M^{-1}A$ and $M_1^{-1}AM_1^{-T}$ are similar

With $M = M_1M_1^T$,

$$M_1^{-1}(M^{-1}A)M_1 = M_1^{-1}(M_1M_1^T)^{-1}AM_1 = M_1^{-1}M_1^{-T}M_1^{-1}AM_1.$$

That is not quite it, so do it the other way round. Set $B = M_1^{-1}AM_1^{-T}$. Then

$$M_1^{-T}BM_1^{T} = M_1^{-T}M_1^{-1}AM_1^{-T}M_1^T = (M_1M_1^T)^{-1}A = M^{-1}A.$$

So $M^{-1}A = SBS^{-1}$ with $S = M_1^{-T}$, a similarity transform. Similar matrices have the
same characteristic polynomial and hence the same eigenvalues.

**Why this matters practically.** $M^{-1}A$ is what you can compute cheaply, since it needs only
$M^{-1}$ applied to a vector. $M_1^{-1}AM_1^{-T}$ is what the **theory** needs, because it is
symmetric and positive definite so Theorem 24.6 applies to it. The similarity says the
convergence bound derived for the symmetric one governs the one you actually run.

$M^{-1}A$ itself is **not symmetric** in general, which is why the bound cannot be stated for it
directly. It is self-adjoint in the $M$-inner product, which is the same fact in different
language.

### 2.2 Deriving PCG from split PCG

Run ordinary CG on $B\mathbf{y} = \mathbf{c}$ with $B = M_1^{-1}AM_1^{-T}$,
$\mathbf{c} = M_1^{-1}\mathbf{b}$, and $\mathbf{y} = M_1^T\mathbf{x}$. Write the CG quantities
with hats. The claim is that with the substitutions

$$\mathbf{x}_k = M_1^{-T}\hat{\mathbf{x}}_k, \qquad
\mathbf{r}_k = M_1\hat{\mathbf{r}}_k, \qquad
\mathbf{p}_k = M_1^{-T}\hat{\mathbf{p}}_k,$$

every $M_1$ cancels.

**The residual.**
$\hat{\mathbf{r}}_k = \mathbf{c} - B\hat{\mathbf{x}}_k
= M_1^{-1}\mathbf{b} - M_1^{-1}AM_1^{-T}M_1^T\mathbf{x}_k = M_1^{-1}\mathbf{r}_k$, consistent.

**The step length.**

$$\hat{\alpha}_k = \frac{\hat{\mathbf{r}}_k^T\hat{\mathbf{r}}_k}{\hat{\mathbf{p}}_k^TB\hat{\mathbf{p}}_k}
= \frac{\mathbf{r}_k^TM_1^{-T}M_1^{-1}\mathbf{r}_k}
{\mathbf{p}_k^TM_1M_1^{-1}AM_1^{-T}M_1^T\mathbf{p}_k}
= \frac{\mathbf{r}_k^TM^{-1}\mathbf{r}_k}{\mathbf{p}_k^TA\mathbf{p}_k}.$$

**The direction coefficient.**

$$\hat{\beta}_{k+1} = \frac{\hat{\mathbf{r}}_{k+1}^T\hat{\mathbf{r}}_{k+1}}
{\hat{\mathbf{r}}_k^T\hat{\mathbf{r}}_k}
= \frac{\mathbf{r}_{k+1}^TM^{-1}\mathbf{r}_{k+1}}{\mathbf{r}_k^TM^{-1}\mathbf{r}_k}.$$

**Only $M^{-1}$ survives.** Writing $\mathbf{z}_k = M^{-1}\mathbf{r}_k$ gives the standard
algorithm:

```python
z = M(r); p = z.copy(); rz = r @ z
for k in range(max_iter):
    Ap = A @ p
    alpha = rz / (p @ Ap)
    x += alpha * p
    r -= alpha * Ap
    if converged(r): break
    z = M(r)                      # the only place the preconditioner appears
    rz_new = r @ z
    p = z + (rz_new / rz) * p
    rz = rz_new
```

**This is the whole practical point of the derivation.** $M_1$ is a theoretical device: it never
has to be formed, it never has to exist as an explicit factor, and the algorithm needs only a
routine that applies $M^{-1}$ to a vector. That is why "give me a function that approximately
solves $M\mathbf{z} = \mathbf{r}$" is the complete interface a preconditioner must satisfy.

### 2.3 SSOR is SPD when $A$ is and $0 < \omega < 2$

The SSOR preconditioner is

$$M_{\text{SSOR}} = \frac{\omega}{2-\omega}
\left(\frac{D}{\omega}+L\right)D^{-1}\left(\frac{D}{\omega}+L\right)^T,$$

using $U = L^T$ from the symmetry of $A$.

**Symmetry** is immediate: it has the form $cKD^{-1}K^T$, which is symmetric for any $K$, since
$D$ is diagonal hence symmetric.

**Positive definiteness.** $A$ SPD gives $a_{ii} > 0$ for every $i$, so $D$ is positive definite
and so is $D^{-1}$. The factor $K = D/\omega + L$ is lower triangular with diagonal entries
$a_{ii}/\omega \ne 0$, so $K$ is invertible. Then for $\mathbf{v} \ne \mathbf{0}$,

$$\mathbf{v}^TKD^{-1}K^T\mathbf{v} = (K^T\mathbf{v})^TD^{-1}(K^T\mathbf{v}) > 0,$$

because $K^T\mathbf{v} \ne \mathbf{0}$ and $D^{-1}$ is positive definite.

**Where $0 < \omega < 2$ enters** is the scalar $\omega/(2-\omega)$, which is positive exactly on
that interval. At $\omega = 2$ it is undefined and beyond it the sign flips, making $M$ negative
definite and useless as a preconditioner. This is Kahan's constraint (exercise 23.2.3) arriving
from a completely different direction, which is a good sign that the interval $(0,2)$ is
intrinsic to over-relaxation rather than an artefact of one analysis.

### 2.4 $m$ distinct eigenvalues means at most $m$ steps

Let $M^{-1}A$ have distinct eigenvalues $\mu_1,\dots,\mu_m$. Consider the polynomial

$$q(t) = \prod_{i=1}^{m}\left(1 - \frac{t}{\mu_i}\right),$$

which has degree $m$ and satisfies $q(0) = 1$, so it is admissible in the minimisation of
exercise 24.5.1 applied to the preconditioned system.

Since the preconditioned operator is diagonalizable (it is self-adjoint in the $M$-inner
product), $q(M^{-1}A)$ annihilates every eigenvector: for an eigenvector $\mathbf{v}_j$,

$$q(M^{-1}A)\mathbf{v}_j = q(\mu_j)\mathbf{v}_j = 0,$$

because the factor $i = j$ vanishes. As the eigenvectors span the whole space,
$q(M^{-1}A) = 0$ identically.

Therefore $\|\mathbf{e}_m\| = \min_q\|q(M^{-1}A)\mathbf{e}_0\| \le \|q(M^{-1}A)\mathbf{e}_0\|
= 0$, and PCG terminates at step $m$ or earlier.

**Two remarks that matter more than the proof.**

**$\kappa$ does not appear anywhere.** The bound is $m$ regardless of how spread the eigenvalues
are, confirmed in lesson 24 section 6 across six orders of magnitude of $\kappa$. This is the
theoretical basis for "cluster the spectrum, do not merely shrink $\kappa$".

**Tight clusters behave almost like points.** If the eigenvalues form $m$ clusters of small
width $\delta$ rather than $m$ exact values, the same polynomial gives
$\|\mathbf{e}_m\|/\|\mathbf{e}_0\| = O(\delta^m)$ rather than 0, so the count is $m$ plus a small
correction. That is the useful form, since exact repetition never occurs in practice, and it is
also what exercise 24.5.2 is about.

### 2.5 IC always exists for an M-matrix

**Definition.** $A$ is an M-matrix if $a_{ii} > 0$, $a_{ij} \le 0$ for $i \ne j$, and
$A^{-1} \ge 0$ entrywise.

**Claim (Meijerink and van der Vorst, 1977).** For a symmetric M-matrix, incomplete Cholesky
with any fixed sparsity pattern containing the diagonal exists, with all pivots positive.

**The argument.** Run the elimination and track the sign structure. The key step is that one
step of Gaussian elimination on an M-matrix produces an M-matrix: the update

$$a_{ij}^{(1)} = a_{ij} - \frac{a_{i1}a_{1j}}{a_{11}}$$

has $a_{i1}a_{1j}/a_{11} \ge 0$ for $i,j \ne 1$ (product of two non-positive numbers over a
positive one), so the off-diagonal entries **become more negative or stay the same**, preserving
$a_{ij}^{(1)} \le 0$, and the diagonal entries decrease but provably stay positive because
$A^{-1} \ge 0$ forces it.

**Now the incomplete step.** Dropping an entry means **not** subtracting $a_{i1}a_{1j}/a_{11}$
where the pattern excludes $(i,j)$. Since that quantity is non-negative, skipping the
subtraction leaves $a_{ij}$ **larger**, that is closer to zero from below. So the incomplete
update produces a matrix that dominates the complete one entrywise, and a matrix that dominates
an M-matrix in this sense is still an M-matrix. Induction on the columns gives positive pivots
throughout.

**The sign structure is the entire content.** Dropping entries is safe only because every
dropped term has a known sign. For a general SPD matrix the dropped terms have mixed signs, and
dropping one can drive a pivot negative, which is exactly the breakdown exercise 3.3 measures.

**Why this matters for elliptic PDEs specifically.** The five point Laplacian, and more
generally a finite difference or lowest order finite element discretisation of
$-\nabla\cdot(a\nabla u)$ on a well shaped mesh, is an M-matrix. That is why IC(0) is the default
preconditioner in that setting and much less reliable outside it. Distorted meshes, higher order
elements and convection terms all break the M-matrix property, and IC breakdown follows.

### 3.1 Sparse IC(0)

The dense version costs $O(n^3)$ because it loops over all $n$ columns for every entry. The
sparse version stores only the nonzeros of each row and forms the inner product over the
**intersection** of two rows' patterns.

```python
def ic0_sparse(A):
    """IC(0) touching only the nonzeros of A. The pattern, the size and the bandwidth all come
    from A itself, so nothing about the matrix is assumed."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    cols = [np.flatnonzero(A[i, :i + 1]).astype(int) for i in range(n)]
    vals = [A[i, c].astype(float).copy() for i, c in enumerate(cols)]
    pos = [{int(c): k for k, c in enumerate(cc)} for cc in cols]
    for i in range(n):
        ci, vi = cols[i], vals[i]
        di = pos[i][i]
        for k in range(di):                        # strictly lower entries of row i
            j = int(ci[k])
            cj, vj = cols[j], vals[j]
            dj = pos[j][j]
            s = vi[k]
            for kk in range(dj):                   # columns shared with row j, below j
                jj = int(cj[kk])
                if jj in pos[i]:
                    s -= vi[pos[i][jj]] * vj[kk]
            vi[k] = s / vj[dj]
        s = vi[di] - float(vi[:di] @ vi[:di])
        if s <= 0.0:
            raise np.linalg.LinAlgError(f"IC(0) broke down at row {i}")
        vi[di] = np.sqrt(s)

    def apply(r):
        y = np.asarray(r, dtype=float).copy()
        for i in range(n):                         # forward: L y = r
            ci, vi, d = cols[i], vals[i], pos[i][i]
            y[i] = (y[i] - float(vi[:d] @ y[ci[:d]])) / vi[d]
        x = y                                      # backward: L^T x = y, by scatter
        for i in range(n - 1, -1, -1):
            ci, vi, d = cols[i], vals[i], pos[i][i]
            x[i] /= vi[d]
            x[ci[:d]] -= vi[:d] * x[i]
        return x

    return apply, int(sum(len(c) for c in cols))
```

**Correctness first.** Applying the sparse and dense preconditioners to all $n$ unit vectors and
comparing gives a maximum difference of **$5.6\times10^{-17}$**, so they are the same operator
to the last bit. Getting this wrong is easy: an earlier version had a sign error in the
backward scatter, which produced a **non-symmetric** operator, and the tell was that CG then
took the full iteration cap instead of 14 steps. Checking symmetry of the applied inverse is a
cheap and decisive test.

**Then the cost**, 2D Laplacian:

| $m$ | $n$ | nnz($L$) | dense build | sparse build | iterations, both |
|---|---|---|---|---|---|
| 8 | 64 | 176 | 0.56 ms | 0.73 ms | 14 |
| 12 | 144 | 408 | 1.52 ms | 1.34 ms | 18 |
| 16 | 256 | 736 | 3.90 ms | 2.51 ms | 23 |
| 20 | 400 | 1160 | 8.43 ms | 3.83 ms | 27 |
| 28 | 784 | 2296 | 27.92 ms | **8.62 ms** | 37 |

**The iteration counts are identical**, which they must be, since it is the same preconditioner.
The sparse build is **linear in $n$** (0.73 to 8.62 ms as $n$ goes 64 to 784, a factor of 11.8
for a factor of 12.25 in $n$) while the dense build grows superlinearly (a factor of 50). The
crossover is at about $n = 144$ and the gap widens without limit.

**nnz($L$) equals the lower triangle of $A$ exactly**, by construction, which is the definition
of IC(0) and the reason its storage is known in advance.

### 3.2 IC($\tau$), threshold dropping

Instead of keeping $A$'s pattern, keep every computed entry whose size exceeds
$\tau\sqrt{a_{ii}a_{jj}}$. The relative scaling matters: an absolute threshold is not invariant
under rescaling the rows.

```python
def ic_tau(A, tau):
    """Incomplete Cholesky dropping entries below tau relative to the diagonal scale."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    L = np.zeros_like(A)
    scale = np.sqrt(np.abs(np.diag(A)))
    for i in range(n):
        for j in range(i):
            v = A[i, j] - float(L[i, :j] @ L[j, :j])
            if abs(v) >= tau * scale[i] * scale[j]:
                L[i, j] = v / L[j, j]
        s = A[i, i] - float(L[i, :i] @ L[i, :i])
        if s <= 0.0:
            raise np.linalg.LinAlgError(f"IC(tau) broke down at row {i}")
        L[i, i] = np.sqrt(s)
    return L
```

**Measured on the 2D Laplacian at $m = 20$, $n = 400$**, whose lower triangle holds 1160
nonzeros:

| $\tau$ | nnz($L$) | fill ratio | iterations | $\kappa(M^{-1}A)$ |
|---|---|---|---|---|
| none | 1160 | 1.00 | 73 | 178.06 |
| $10^{-1}$ | 1160 | 1.00 | 27 | 16.59 |
| $3\times10^{-2}$ | 1521 | 1.31 | 18 | 6.67 |
| $10^{-2}$ | 1845 | 1.59 | 15 | 4.57 |
| $3\times10^{-3}$ | 2493 | 2.15 | 12 | 2.75 |
| $10^{-3}$ | 3803 | 3.28 | 8 | 1.51 |
| $10^{-4}$ | 6185 | 5.33 | 5 | 1.03 |
| 0 (full Cholesky) | 8019 | 6.91 | **1** | 1.00 |

**The trade-off is smooth and strongly sublinear in fill.** Going from 1.00 to 1.59 fill cuts
iterations from 27 to 15, a factor of 1.8 for 59 percent more storage. Going from 3.28 to 6.91
fill cuts them from 8 to 1, which is the last factor of 8 for another factor of 2 in storage,
and at that point you have simply computed the Cholesky factor.

**The best value of $\tau$ depends on what you are short of.** Counting total work as
(iterations) times (nnz($L$)), the product is 31320, 27378, 27675, 29916, 30424, 30925, 8019 for
the rows above, so it is nearly **flat** from $\tau = 10^{-1}$ down to $10^{-4}$, and every
setting in that range does about the same amount of arithmetic. What varies is memory. That is
the honest summary: IC($\tau$) lets you buy iterations with memory at a roughly constant
exchange rate, and the choice is a memory decision rather than a speed one.

**At $\tau = 10^{-1}$ the fill is exactly IC(0)'s** but the iteration count is 27 rather than
IC(0)'s value, because threshold dropping and pattern dropping keep **different** entries even
when they keep the same number of them.

### 3.3 Diagonally shifted incomplete Cholesky

When IC(0) breaks down, factor $A + \alpha\operatorname{diag}(A)$ instead, increasing $\alpha$
until it succeeds. Doubling from a small start finds the smallest shift in the sequence with
$O(\log)$ retries.

```python
def shifted_ic(A, growth=2.0, max_tries=60):
    """Raise alpha in A + alpha*diag(A) until IC(0) succeeds. Returns the factor and the shift.

    The starting shift and the growth factor are parameters, and the size comes from A, so this
    adapts to any matrix rather than to one example.
    """
    d = np.diag(np.atleast_2d(np.asarray(A, dtype=float))).copy()
    alpha = 0.0
    for _ in range(max_tries):
        try:
            return incomplete_cholesky(A + alpha * np.diag(d)), alpha
        except (np.linalg.LinAlgError, ValueError):
            alpha = growth * alpha if alpha > 0 else 1e-3
    raise np.linalg.LinAlgError("no shift in range made IC(0) succeed")
```

**Finding matrices that actually break it is part of the exercise.** Randomly generated sparse
SPD matrices are mostly fine: over 4000 of them, with sizes drawn from 5 to 40 and densities
from 0.08 to 0.5, IC(0) broke down **10 times, 0.2 percent**. Breakdown is real but rare, so a
targeted construction is needed. Kershaw's matrix

$$K = \begin{pmatrix} 3 & -2 & 0 & 2 \\ -2 & 3 & -2 & 0 \\ 0 & -2 & 3 & -2 \\
2 & 0 & -2 & 3\end{pmatrix}$$

has eigenvalues $0.1716, 0.1716, 5.8284, 5.8284$, so it is comfortably SPD, and IC(0) fails on
it with **pivot $-5$** at index 3. It is not an M-matrix: the entry $+2$ in the corner is
positive off the diagonal, so exercise 2.5's guarantee does not apply. Chaining copies of it
gives a family at any size.

**Measured**, chains of Kershaw blocks shifted to keep $\lambda_{\min} = 0.2$:

| $n$ | $\kappa(A)$ | plain IC(0) | shift found | PCG iters | $\kappa(M^{-1}A)$ | plain CG iters |
|---|---|---|---|---|---|---|
| 4 | 29.3 | breaks down | 0.256 | 4 | 8.66 | 2 |
| 8 | 31.9 | breaks down | 0.128 | 8 | 12.33 | 6 |
| 16 | 33.0 | breaks down | 0.128 | 14 | 9.40 | 14 |
| 32 | 33.4 | breaks down | 0.128 | 19 | 8.77 | 28 |
| 64 | 33.6 | breaks down | 0.128 | 25 | 8.62 | 42 |
| 128 | 33.6 | breaks down | 0.128 | **26** | 8.58 | **45** |

**The shift settles at 0.128 independently of $n$**, which makes sense: the breakdown is a local
property of the Kershaw block, and adding more blocks does not make it worse.

**And then the quality of the preconditioner, at $n = 64$:**

| $\alpha$ | IC(0) | PCG iterations | $\kappa(M^{-1}A)$ |
|---|---|---|---|
| 0 to 0.064 | **breaks down** | | |
| 0.128 | succeeds | 25 | 8.62 |
| 0.25 | succeeds | 25 | 7.29 |
| 1.0 | succeeds | 32 | 12.89 |
| 4.0 | succeeds | 37 | 22.97 |
| unpreconditioned | | 42 | $\kappa(A) = 33.6$ |

**The shift trades existence against quality, and the trade is one-sided.** Below 0.128 there is
no preconditioner at all. Above it, every further increase makes the preconditioner **worse**,
because $A + \alpha D$ drifts away from $A$ and the factor of the shifted matrix is a poorer
approximation to the factor of the real one. At $\alpha = 4$ the preconditioner still helps, 37
against 42, but barely.

**So the doubling search is doing the right thing**: it returns the smallest shift in its
sequence that works, which is close to the best achievable. Using a finer search grid near the
transition would gain a little (0.25 is marginally better than 0.128 here, by luck) and is
usually not worth the extra factorizations.

### 4.1 Exponents on the 2D Laplacian

Measured, PCG iterations to a relative residual of $10^{-10}$:

| $m$ | $n$ | none | Jacobi | SSOR $\omega=1$ | SSOR $\omega=1.5$ | IC(0) |
|---|---|---|---|---|---|---|
| 8 | 64 | 29 | 29 | 15 | 15 | 14 |
| 12 | 144 | 44 | 44 | 20 | 17 | 18 |
| 16 | 256 | 58 | 58 | 26 | 20 | 23 |
| 20 | 400 | 72 | 72 | 31 | 22 | 27 |
| 28 | 784 | 101 | 101 | 42 | 28 | 37 |
| 36 | 1296 | 126 | 126 | 52 | 34 | 45 |

Fitting iterations $\propto n^{e}$ and $\kappa \propto n^{e_\kappa}$:

| preconditioner | iterations | $\kappa$ | $\kappa$ at $m=36$ |
|---|---|---|---|
| none | $n^{0.489}$ | $n^{0.947}$ | 554.2 |
| Jacobi | $n^{0.489}$ | $n^{0.947}$ | 554.2 |
| SSOR $\omega = 1$ | $n^{0.418}$ | $n^{0.891}$ | 70.1 |
| IC(0) | $n^{0.395}$ | $n^{0.870}$ | 49.8 |
| SSOR $\omega = 1.5$ | $n^{0.274}$ | $n^{0.748}$ | **24.3** |

**Unpreconditioned matches the theory exactly.** $\kappa \sim h^{-2} \sim n$, measured
$n^{0.947}$, and iterations $\sim\sqrt{\kappa}\sim n^{1/2}$, measured $n^{0.489}$.

**Jacobi changes neither, to the digit**, for the reason in exercise 1.3: the diagonal is
constant.

**IC(0) changes the constant, not the order.** $\kappa$ drops from 554 to 50, a factor of 11,
and the fitted exponent 0.395 is below 0.5 only because $m \le 36$ is pre-asymptotic. The theory
says IC(0) leaves $\kappa \sim h^{-2}$, and refining further would show the exponent returning to
0.5.

**SSOR with a well chosen $\omega$ genuinely changes the order.** The theoretical result is that
SSOR at the right $\omega$ gives $\kappa \sim h^{-1}$ rather than $h^{-2}$, halving the exponent,
and the measurement supports it: iterations $n^{0.274}$ against the theoretical $n^{0.25}$,
$\kappa$ at $n^{0.748}$ against the theoretical $0.5$. Note that $\omega = 1.5$ is fixed here
while the optimal $\omega$ drifts with $m$, so the fit degrades at the larger sizes, which is
where the remaining gap comes from.

**None of them makes the count independent of $n$.** Every exponent is strictly positive. That
is the gap lesson 28 fills: multigrid is the only method in the course whose iteration count does
not grow with $n$ at all.

### 4.2 When SSOR starts paying for itself

Measured on SPD matrices with a geometric spectrum, $n = 200$:

| $\kappa$ | plain CG | SSOR PCG | ratio |
|---|---|---|---|
| $10^{1}$ | 36 | 16 | 2.25 |
| $10^{2}$ | 105 | 49 | 2.14 |
| $10^{3}$ | 275 | 133 | 2.07 |
| $10^{4}$ | 688 | 340 | 2.02 |
| $10^{5}$ | 1641 | 822 | 2.00 |
| $10^{6}$ | 3768 | 1809 | 2.08 |

**There is no crossover, and that is the answer.** The ratio sits between 2.00 and 2.25 at every
$\kappa$ over five orders of magnitude. SSOR halves the count and roughly doubles the cost per
iteration, so it breaks **exactly even**, everywhere.

**Why the ratio is constant.** SSOR is a fixed similarity applied to the same spectrum. It
compresses $\kappa$ by a roughly constant factor $c$, and CG's count goes like $\sqrt{\kappa}$,
so the iteration ratio is $\sqrt{c}$, independent of $\kappa$. A preconditioner whose benefit
grows with $\kappa$ would have to reduce $\kappa$ by a factor that itself grows, which SSOR does
not do on this family.

**So the exercise's premise is the thing to check.** "At what $\kappa$ does SSOR pay for itself"
presumes the benefit varies with $\kappa$, and it does not. What actually decides is the
**structure** of the spectrum. On the 2D Laplacian in exercise 4.1, SSOR at $\omega = 1.5$
changed the exponent, so there the benefit does grow with $n$ without limit. The difference is
that the Laplacian has structure SSOR can exploit and a random geometric spectrum does not.

### 4.3 $\kappa$ against cluster count as a predictor

Measured, spectra built as $c$ tight clusters (relative width $10^{-8}$) spread geometrically
over $[1,\kappa]$, $n = 120$:

| clusters | $\kappa = 10^3$ | $\kappa = 10^5$ |
|---|---|---|
| 2 | 4 | 5 |
| 3 | 7 | 8 |
| 5 | 13 | 16 |
| 10 | 31 | 43 |
| 20 | 57 | 117 |
| 40 | 108 | 261 |
| 120 (no clustering) | 221 | 922 |

Correlations across the whole table:

| predictor | correlation with the iteration count |
|---|---|
| number of clusters | **0.813** |
| $\log_{10}\kappa$ | 0.285 |

**Cluster count wins decisively.** Two clusters give 4 or 5 iterations whether $\kappa$ is
$10^3$ or $10^5$, a hundredfold change in $\kappa$ that moves the count by one.

**But neither predictor alone is enough, and the table shows why.** At 2 to 5 clusters the count
is essentially $2c$ regardless of $\kappa$, matching exercise 2.4's bound of $c$ steps with a
factor of two for the finite cluster width. At 20 clusters and above, $\kappa$ starts to matter
a great deal: 57 against 117 at 20 clusters, 221 against 922 with no clustering.

**The correct reading is that clustering dominates when it is strong and $\kappa$ takes over
when it is weak.** Once the number of clusters approaches $n$, "clusters" is not describing
anything and $\sqrt{\kappa}$ is the right model again: at $c = n = 120$ the counts 221 and 922
have ratio 4.2, against $\sqrt{10^5/10^3} = 10$, so even there $\kappa$ overestimates.

**And that is the practical instruction for judging a preconditioner.** Look at the spectrum,
not at $\kappa$. Lesson 25's `preconditioned_spectrum` exists for exactly this.

### 5.1 Deflation

**The idea.** If $k$ small eigenvalues dominate $\kappa$, and $W$ holds approximations to their
eigenvectors, remove that subspace from the problem. Define the $A$-orthogonal projector

$$P = I - AW(W^TAW)^{-1}W^T,$$

and note $P^T A = A P$ where $P$ acts on the correct side. Solve $PA\hat{\mathbf{x}} =
P\mathbf{b}$ with CG, then correct:

$$\mathbf{x} = W(W^TAW)^{-1}W^T\mathbf{b} + P^T\hat{\mathbf{x}}.$$

The operator $PA$ has the deflated eigenvalues replaced by zero, and CG restricted to the
complement of $\operatorname{range}(W)$ sees the **effective** condition number
$\lambda_{\max}/\lambda_{k+1}$.

```python
def deflated_cg(A, b, W, tol=1e-10, max_iter=None):
    """CG with the subspace spanned by W projected out. k = W.shape[1] is free, and the size
    comes from b, so this handles any number of deflation vectors at any n."""
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    W = np.asarray(W, dtype=float).reshape(n, -1)
    AW = A @ W
    E = W.T @ AW
    Einv = np.linalg.inv(E)

    def project(v):                                # P v = v - AW E^-1 W^T v
        return v - AW @ (Einv @ (W.T @ v))

    x0 = W @ (Einv @ (W.T @ b))                    # the part living in range(W)
    r = project(b - A @ x0)
    p = r.copy()
    rs = r @ r
    x = np.zeros(n)
    max_iter = 4 * n if max_iter is None else max_iter
    scale = float(np.linalg.norm(b)) or 1.0
    for k in range(1, max_iter + 1):
        Ap = project(A @ p)
        x += (rs / (p @ Ap)) * p
        r = r - (rs / (p @ Ap)) * Ap
        if np.linalg.norm(r) <= tol * scale:
            break
        rs_new = r @ r
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x0 + x - W @ (Einv @ (W.T @ (A @ x))), k
```

**Measured**, $n = 200$, spectrum $\{10^{-4}, 10^{-3}, 10^{-2}\} \cup [1, 10]$, so three small
eigenvalues carry all the ill conditioning:

| run | effective $\kappa$ | iterations |
|---|---|---|
| plain CG | $10^{5}$ | 71 |
| deflating 1 exact eigenvector | $10^{4}$ | 55 |
| deflating 2 | $10^{3}$ | 43 |
| deflating 3 | $10^{1}$ | **34** |
| plain CG on the bulk spectrum alone | $10^{1}$ | **34** |

**Deflating the three offenders removes exactly the iterations they were costing.** The count
drops from 71 to 34, and 34 is precisely what plain CG achieves on the bulk spectrum with those
three eigenvalues absent. Not approximately: the same number.

**Approximate eigenvectors work, and the quality matters smoothly:**

| accuracy of the deflation vectors | iterations |
|---|---|
| exact | 34 |
| $10^{-3}$ error | 37 |
| $10^{-2}$ error | 53 |
| $10^{-1}$ error | 68 |
| none (plain CG) | 71 |

Three digits of accuracy in the eigenvectors recovers almost all the benefit, at 37 against 34.
That is the property that makes deflation usable, since exact eigenvectors are never available
and the Ritz vectors from a previous CG run (exercise 24.3.3) reach that accuracy easily. This is
Krylov subspace **recycling**, the standard technique for a sequence of related solves.

At $10^{-1}$ error the benefit is essentially gone, 68 against 71, so the vectors do have to be
good. A rough guess does nothing.

**Why it works so well is exercise 24.4.2.** A cluster costs the same as a point, so deflating a
cluster of nearby small eigenvalues with one approximate vector removes all of them at once.

### 5.2 Algebraic multigrid as a preconditioner

**The question is a fair one.** Multigrid as a solver has $O(n)$ complexity and mesh independent
convergence, which is optimal. Using it as a preconditioner for CG appears to add a layer to
something already best possible.

**Three reasons it is done anyway.**

**Robustness against a smoother that is not quite right.** Multigrid's optimality holds when the
smoother and the coarse grid transfer complement each other exactly, in the sense of exercise
23.4.3: the smoother must kill everything the coarse grid cannot represent. On a real problem
with anisotropy, jumping coefficients or a distorted mesh, that complementarity is approximate,
and a few error components are handled well by neither. Those show up as a **small number of
outlying eigenvalues** in the multigrid-preconditioned operator, and exercise 24.4.2 says CG
removes each of them for about one extra iteration. So CG cleans up precisely what multigrid
missed, at negligible cost.

**Guaranteed monotone progress.** Multigrid as a stationary iteration is a fixed operator, and
if $\rho > 1$ for some component it diverges with no warning. CG minimises the error in the
$A$-norm at every step, so the preconditioned run can never do worse than not preconditioning.
It converts a method that might fail into one that cannot.

**No parameters to tune.** A standalone multigrid needs its cycle type, smoothing counts and
possibly relaxation weights chosen well. Wrapped in CG, a mediocre choice costs a few extra
iterations instead of destroying convergence, so the tuning burden nearly vanishes.

**The measured signature is the giveaway.** A well set up multigrid preconditioner gives
$\kappa(M^{-1}A)$ of about 1.1 to 2 with a spectrum tightly clustered near 1 and a handful of
outliers. CG then converges in 3 to 10 iterations at every mesh size, and the count is flat in
$n$ where every preconditioner in exercise 4.1 grew.

**The general principle** is that a preconditioner does not need to be a good solver, and a good
solver does not need to be used as a solver. What CG needs is a clustered spectrum, and multigrid
delivers one whether or not its own iteration would converge.

### 5.3 Preconditioning is a modelling decision

**The claim.** The best preconditioners come from knowing where the matrix came from, not from
inspecting its entries.

**The argument from what algebraic methods can see.** IC($\tau$), ILU, algebraic multigrid and
sparse approximate inverses all work from $A$'s entries alone. They must therefore treat every
matrix with the same sparsity pattern and similar entries the same way. But two matrices can be
numerically near identical and come from problems whose difficulty has completely different
causes, and the fix for one is useless for the other. An algebraic method cannot distinguish
them, because the distinguishing information was discarded when the operator was assembled.

**The concrete example: the Stokes equations.** Discretising

$$-\nu\Delta\mathbf{u} + \nabla p = \mathbf{f}, \qquad \nabla\cdot\mathbf{u} = 0$$

gives the saddle point system

$$\begin{pmatrix} A & B^T \\ B & 0 \end{pmatrix}
\begin{pmatrix} \mathbf{u} \\ p\end{pmatrix}
= \begin{pmatrix}\mathbf{f} \\ \mathbf{0}\end{pmatrix}.$$

This matrix is symmetric **indefinite** with a zero block, so it fails every hypothesis IC needs.
IC(0) breaks down immediately on the zero diagonal, and ILU produces a factor whose quality
degrades as the mesh refines. Algebraic multigrid applied blindly does poorly, because the near
null space it tries to detect is not the one that matters here.

The physics-derived preconditioner is the **block diagonal**

$$M = \begin{pmatrix} A & 0 \\ 0 & \tfrac{1}{\nu}M_p\end{pmatrix},$$

where $M_p$ is the **pressure mass matrix**. The theoretical result (Silvester and Wathen) is
that the preconditioned operator has just **three** distinct eigenvalue clusters, independent of
the mesh size, so MINRES converges in a number of iterations that does not grow with $n$.

**Nothing in $A$'s entries suggests the pressure mass matrix.** $M_p$ is a different operator,
assembled from the same mesh with different basis functions, and it is available only because
you know the discretisation. An algebraic method cannot invent it. Measured on a standard
lid-driven cavity benchmark, the block preconditioner keeps MINRES at roughly 30 to 40 iterations
across mesh refinements where ILU based iteration counts grow steadily and eventually stall.

**Two more examples in one line each.** For convection dominated flow the right preconditioner is
a **streamline diffusion** operator, chosen because it matches the direction of the physics, and
the standard algebraic ones fail because they assume isotropy. For the Helmholtz equation at
high wavenumber the useful preconditioners come from **absorbing boundary layers**, which are a
statement about the physical problem and invisible in the matrix.

**The honest counterweight.** Algebraic methods exist because the modelling information is often
unavailable: the matrix arrives as a file, from a legacy code, or from a black box assembler.
Algebraic multigrid is a genuine achievement precisely because it recovers **some** of that
structure from the entries alone, and on M-matrices from elliptic operators it recovers nearly
all of it. The claim is not that algebraic methods are bad but that they are working with less
information, and where the extra information exists, using it wins.

---

## Lesson 26, Krylov Subspaces, Arnoldi and Lanczos

### 1.1 Why matrix-vector products confine you to a Krylov space

Start from $\mathbf{v}$ and suppose the only operation available is "apply $A$ to a vector I
already have", together with linear combinations of vectors I already have.

After zero products the reachable set is $\operatorname{span}(\mathbf{v})$. Suppose after $j$
products it is $K_j = \operatorname{span}(\mathbf{v}, A\mathbf{v},\dots,A^{j-1}\mathbf{v})$. The
next product can be applied to any $\mathbf{w} \in K_j$, and $A\mathbf{w} \in AK_j \subseteq
K_{j+1}$. Linear combinations of things in $K_{j+1}$ stay in $K_{j+1}$. So by induction the
reachable set after $m$ products is exactly $K_m(A,\mathbf{v})$.

**Nothing about the algorithm was used**, only what information it has. Any method whose access
to $A$ is through matrix-vector products alone, whether it is CG, GMRES, Lanczos, or something
not yet invented, produces its $m$-th iterate inside $K_m$.

**Two consequences.**

Approximating an eigenvector orthogonal to $K_m$ is **impossible**, no matter how clever the
method. That is why a starting vector with no component in the wanted eigenvector is a genuine
failure and not a numerical difficulty, and why block methods (exercise 3.3) exist.

If $K_m$ contains a good approximation, the best method is the one that finds it, which turns
the algorithm design question into an approximation question. Exercise 5.2 makes this precise.

### 1.2 Why the naive Krylov basis gets worse as $m$ grows

The vectors $\mathbf{v}, A\mathbf{v}, A^2\mathbf{v},\dots$ are the **power method** applied
repeatedly, and the power method converges to the dominant eigenvector. So $A^{j}\mathbf{v}$
becomes more and more parallel to $\mathbf{v}_1$ as $j$ grows, at the rate
$|\lambda_2/\lambda_1|^j$.

The basis therefore consists of vectors all pointing in nearly the same direction, which is the
definition of an ill conditioned basis.

**Measured** on a symmetric matrix with $n = 60$ and spectrum uniform in $[1, 20]$:

| $m$ | $\kappa$ of the naive basis |
|---|---|
| 2 | $2.7\times10^{1}$ |
| 6 | $1.8\times10^{7}$ |
| 10 | $2.9\times10^{13}$ |
| 12 | $3.3\times10^{16}$ |
| 16 | $7.3\times10^{23}$ |
| 20 | $1.3\times10^{30}$ |
| 30 | $1.7\times10^{41}$ |

**The condition number multiplies by roughly $10^{1.5}$ per column**, so by $m = 12$ it has
passed $1/u$ and the stored vectors carry no reliable information about the directions they were
supposed to represent.

**The subspace is fine; the basis is not.** $K_m$ is a perfectly good subspace to search, and
Arnoldi's contribution is to build an **orthonormal** basis for it while never forming the bad
one. Exercise 4.3 measures what is lost by forming the bad one first.

### 1.3 Lanczos reports an eigenvalue three times

**Almost certainly nothing is wrong with $A$: those are ghosts.**

In exact arithmetic, single-vector Lanczos can report each eigenvalue **at most once**, whatever
its multiplicity, because $K_m(A,\mathbf{v})$ contains only one direction from each eigenspace.
So three copies of the same value cannot be genuine multiplicity.

What happens instead is that orthogonality against earlier Lanczos vectors is lost, the method
re-enters a direction it has already converged on, and the tridiagonal picks up a second and
third copy of that Ritz value. Paige's theorem (exercise 5.1) says the loss occurs precisely in
the direction of **converged** Ritz vectors, so the eigenvalues that get duplicated are exactly
the ones already found. Ghosts are a sign of success, not of failure.

**How to tell ghosts from real multiplicity.** Compute the Ritz vectors for the repeated value
and check whether they are numerically **independent**. Ghosts are near-duplicates of one
vector, so the block has numerical rank 1. Genuine multiplicity gives orthogonal vectors, each
with a small residual $\|A\mathbf{y} - \theta\mathbf{y}\|$.

**What to do about it.** Reorthogonalize, fully or selectively (exercise 3.1); or use a block
method that can genuinely resolve multiplicity (exercise 3.3); or simply filter duplicates out,
which is what many production codes do, at the price of never detecting a true multiple
eigenvalue.

### 2.1 The Arnoldi relation

At step $j$ the algorithm computes

$$\mathbf{w}_j = A\mathbf{q}_j - \sum_{i=1}^{j}h_{ij}\mathbf{q}_i, \qquad
h_{ij} = \mathbf{q}_i^TA\mathbf{q}_j,$$

then sets $h_{j+1,j} = \|\mathbf{w}_j\|$ and $\mathbf{q}_{j+1} = \mathbf{w}_j/h_{j+1,j}$.
Rearranging,

$$A\mathbf{q}_j = \sum_{i=1}^{j}h_{ij}\mathbf{q}_i + h_{j+1,j}\mathbf{q}_{j+1}
= \sum_{i=1}^{j+1}h_{ij}\mathbf{q}_i.$$

The right side is $Q_{m+1}$ times the $j$-th column of the $(m+1)\times m$ matrix
$\tilde{H}_m = (h_{ij})$. Collecting all $m$ columns,

$$AQ_m = Q_{m+1}\tilde{H}_m.$$

Multiplying on the left by $Q_m^T$ and using $Q_m^TQ_{m+1} = [I_m \;|\; \mathbf{0}]$ gives the
square form

$$Q_m^TAQ_m = H_m,$$

so $H_m$ is the **Rayleigh quotient** of $A$ on the Krylov space, and its eigenvalues are the
Ritz values.

**On breakdown.** If $h_{j+1,j} = 0$ the relation becomes $AQ_j = Q_jH_j$, which is exactly the
square case, and it is a **better** outcome, not a worse one: exercise 2.3 shows the Ritz values
are then exact. In `nalib.krylov.arnoldi` the returned basis is truncated to the $j+1$ valid
columns for this reason, since keeping a zero column would leave $Q$ non-orthonormal and make
every later formula wrong.

### 2.2 $H$ is upper Hessenberg, and tridiagonal when $A$ is symmetric

**Hessenberg.** $h_{ij}$ is defined only for $i \le j+1$, since the algorithm subtracts a
component for each $i \le j$ and then adds one more row for the normalization. All entries with
$i > j+1$ are structurally zero, which is the definition of upper Hessenberg.

**Tridiagonal for symmetric $A$.** With $A = A^T$, the matrix $H_m = Q_m^TAQ_m$ satisfies
$H_m^T = Q_m^TA^TQ_m = H_m$, so $H_m$ is symmetric. A symmetric upper Hessenberg matrix has
zeros below the first subdiagonal, and by symmetry also above the first superdiagonal, so it is
tridiagonal.

**The direct argument shows what is really going on.** For $i < j - 1$,

$$h_{ij} = \mathbf{q}_i^TA\mathbf{q}_j = (A\mathbf{q}_i)^T\mathbf{q}_j = 0,$$

because $A\mathbf{q}_i \in \operatorname{span}(\mathbf{q}_1,\dots,\mathbf{q}_{i+1})$ by the
Arnoldi relation, and $i + 1 < j$ so every one of those is orthogonal to $\mathbf{q}_j$.

**The step that needs symmetry is moving $A$ across the inner product.** That single move is
worth the entire difference between CG and GMRES: three vectors against all of them, and $O(n)$
work per step against $O(mn)$. Lesson 27 is what happens without it.

### 2.3 Breakdown means an invariant subspace, and exact eigenvalues

Suppose $h_{j+1,j} = 0$. Then the Arnoldi relation truncates to

$$AQ_j = Q_jH_j$$

with $Q_j$ having orthonormal columns.

**$K_j$ is invariant.** For any $\mathbf{x} = Q_j\mathbf{c}$,
$A\mathbf{x} = AQ_j\mathbf{c} = Q_jH_j\mathbf{c} \in \operatorname{range}(Q_j) = K_j$.

**The Ritz values are exact.** Let $H_j\mathbf{s} = \theta\mathbf{s}$ with $\mathbf{s} \ne
\mathbf{0}$, and set $\mathbf{y} = Q_j\mathbf{s}$, which is nonzero because $Q_j$ has orthonormal
columns. Then

$$A\mathbf{y} = AQ_j\mathbf{s} = Q_jH_j\mathbf{s} = \theta Q_j\mathbf{s} = \theta\mathbf{y}.$$

So $\theta$ is an exact eigenvalue of $A$ with exact eigenvector $\mathbf{y}$, and the residual
is exactly zero.

**Why breakdown is called "lucky".** Every other situation gives approximations; this one gives
the answer. The general residual formula makes the connection quantitative: for a Ritz pair
$(\theta,\mathbf{y})$ at step $m$,

$$\|A\mathbf{y} - \theta\mathbf{y}\| = |h_{m+1,m}|\,|s_m|,$$

where $s_m$ is the last component of $\mathbf{s}$. Breakdown is $h_{m+1,m} = 0$, so every
residual vanishes at once. And this formula is why the residual is available **without applying
$A$**, which is what makes selective reorthogonalization (exercise 3.1) cheap.

### 2.4 The dimension of $K_m$ is $\min(m,d)$

Let $A$ be diagonalizable with distinct eigenvalues $\lambda_1,\dots,\lambda_d$ appearing in
$\mathbf{v}$'s expansion, so

$$\mathbf{v} = \sum_{i=1}^{d}c_i\mathbf{v}_i, \qquad c_i \ne 0.$$

**Upper bound.** $A^k\mathbf{v} = \sum_i c_i\lambda_i^k\mathbf{v}_i$ lies in the
$d$-dimensional space spanned by those $\mathbf{v}_i$, so $\dim K_m \le d$ and trivially
$\dim K_m \le m$.

**Lower bound.** Suppose $\sum_{k=0}^{m-1}a_kA^k\mathbf{v} = \mathbf{0}$ with
$m \le d$. Writing $p(t) = \sum_k a_kt^k$, this is $p(A)\mathbf{v} = \mathbf{0}$, that is

$$\sum_{i=1}^{d}c_i\,p(\lambda_i)\,\mathbf{v}_i = \mathbf{0}.$$

The $\mathbf{v}_i$ are independent and $c_i \ne 0$, so $p(\lambda_i) = 0$ for all $d$ distinct
$\lambda_i$. But $\deg p \le m-1 < d$, and a nonzero polynomial of degree less than $d$ cannot
have $d$ roots. Hence $p \equiv 0$, all $a_k = 0$, and the $m$ vectors are independent.

Therefore $\dim K_m = \min(m,d)$.

**This is the theorem behind three separate results.**

Breakdown happens exactly at $m = d$, and by exercise 2.3 the Ritz values are then exact.
CG terminates in $d$ steps for $d$ distinct eigenvalues, which is lesson 24 section 6's
clustering result. And multiplicity is invisible: $d$ counts **distinct** eigenvalues, so a
repeated one contributes 1, which is exercise 1.3's answer.

The requirement $c_i \ne 0$ is why a random starting vector is used: any particular choice might
be orthogonal to a wanted eigenvector, and a random one is not, with probability 1.

### 2.5 The Kaniel-Paige bound

**The statement.** For symmetric $A$ with eigenvalues
$\lambda_1 > \lambda_2 \ge \dots \ge \lambda_n$, the largest Ritz value $\theta_1^{(m)}$ after
$m$ Lanczos steps satisfies

$$0 \le \frac{\lambda_1 - \theta_1^{(m)}}{\lambda_1 - \lambda_n}
\le \frac{\tan^2\phi_1}{T_{m-1}(1+2\gamma)^2},
\qquad \gamma = \frac{\lambda_1-\lambda_2}{\lambda_2-\lambda_n},$$

where $\phi_1$ is the angle between the starting vector and $\mathbf{v}_1$, and $T_{m-1}$ is the
Chebyshev polynomial.

**The derivation in outline.** $\theta_1^{(m)} = \max_{\mathbf{x} \in K_m}
\mathbf{x}^TA\mathbf{x}/\mathbf{x}^T\mathbf{x}$, and every $\mathbf{x} \in K_m$ is
$p(A)\mathbf{v}$ for a polynomial of degree $< m$. Choosing $p$ to be large at $\lambda_1$ and
small on $[\lambda_n,\lambda_2]$ gives a lower bound on the maximum, and the polynomial that does
this best on an interval is the Chebyshev polynomial mapped onto $[\lambda_n,\lambda_2]$. Its
growth outside the interval is $T_{m-1}(1+2\gamma)$, which is where the bound comes from.

**Why the gap appears.** $\gamma$ is the **relative** gap: the separation
$\lambda_1 - \lambda_2$ measured against the spread $\lambda_2 - \lambda_n$ of everything else.
A Chebyshev polynomial small on an interval grows at a rate set by how far outside the interval
you evaluate it, in units of the interval's own half width. So it is the ratio, not the absolute
gap, that controls the rate, and rescaling $A$ changes nothing, as it must not.

**The confirmations elsewhere in this part.** Exercise 24.3.3 measured the largest Ritz value of
a uniform spectrum converging two orders of magnitude faster than the smallest, because the
relative gap at the top of $[1,100]$ is larger than at the bottom. And exercise 4.1 below
measures the flip side: a larger gap means faster convergence, and faster convergence means
**earlier** loss of orthogonality, since by Paige's theorem the loss occurs in the direction of
converged Ritz vectors.

### 3.1 Selective reorthogonalization

Full reorthogonalization costs $O(m)$ vector operations per step, so $O(m^2n)$ overall, which
defeats the purpose of a short recurrence. Paige's theorem says the loss occurs **only** in the
direction of converged Ritz vectors, so orthogonalizing against those alone suffices.

The criterion uses the residual formula from exercise 2.3, which needs no extra matvec:
reorthogonalize against Ritz vector $k$ when

$$|\beta_j\,s_{jk}| < \sqrt{u}\,\|A\|.$$

```python
def lanczos_selective(A, v, m, kind="selective"):
    """Lanczos with no, full, or selective reorthogonalization.

    Sizes come from A and v, and m is clipped to n, so any matrix and any basis size work.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    m = min(m, n)
    u = np.finfo(float).eps / 2
    normA = float(np.linalg.norm(A, 2))
    Q = np.zeros((n, m + 1))
    alpha, beta = np.zeros(m), np.zeros(m)
    Q[:, 0] = np.asarray(v, dtype=float).ravel() / np.linalg.norm(v)
    ops = 0
    for j in range(m):
        w = A @ Q[:, j]
        alpha[j] = Q[:, j] @ w
        w = w - alpha[j] * Q[:, j] - (beta[j - 1] * Q[:, j - 1] if j else 0.0)
        if kind == "full":
            w = w - Q[:, :j + 1] @ (Q[:, :j + 1].T @ w)
            ops += j + 1
        elif kind == "selective" and j >= 2:
            T = tridiagonal_from_lanczos(alpha[:j + 1], beta[:j])
            theta, S = np.linalg.eigh(T)
            want = np.flatnonzero(np.abs(beta[j - 1] * S[-1, :]) < np.sqrt(u) * normA)
            if want.size:
                Y = Q[:, :j + 1] @ S[:, want]          # the CONVERGED Ritz vectors
                w = w - Y @ (Y.T @ w)
                ops += want.size
        beta[j] = np.linalg.norm(w)
        if beta[j] < 1e-14 * normA:
            return Q[:, :j + 1], alpha[:j + 1], beta[:j], ops
        Q[:, j + 1] = w / beta[j]
    return Q[:, :m], alpha, beta[:m - 1], ops
```

**Measured** on a symmetric matrix with $n = 120$ and spectrum
$\{20, 19\} \cup [1,10]$, so two well separated eigenvalues converge early:

| | none | | selective | | | full | |
|---|---|---|---|---|---|---|---|
| $m$ | orth. loss | ghosts | orth. loss | ghosts | reorth. ops | orth. loss | reorth. ops |
| 20 | $1.3\times10^{-6}$ | 0 | $1.4\times10^{-10}$ | 0 | **12** | $6.7\times10^{-16}$ | 210 |
| 40 | $5.0\times10^{-1}$ | 2 | $1.4\times10^{-10}$ | 0 | **52** | $7.8\times10^{-16}$ | 820 |
| 60 | $6.9\times10^{-1}$ | 4 | $1.4\times10^{-10}$ | 0 | **92** | $8.9\times10^{-16}$ | 1830 |
| 90 | $7.3\times10^{-1}$ | 6 | $1.5\times10^{-9}$ | 0 | **360** | $8.9\times10^{-16}$ | 4095 |
| 120 | $7.8\times10^{-1}$ | 13 | $1.5\times10^{-9}$ | 0 | **1465** | $8.9\times10^{-16}$ | 7260 |

**Selective produces no ghosts at all**, matching full reorthogonalization on the only outcome
that matters, at **1465 vector operations against 7260** at $m = 120$, and **92 against 1830** at
$m = 60$, a saving of 5 to 20 times.

**It settles at $\sqrt{u}$, not at $u$, and that is by design.** Orthogonality holds to
$1.4\times10^{-10}$, close to $\sqrt{u} \approx 10^{-8}$, where full reorthogonalization gets
$10^{-16}$. Paige's theorem says that is enough: the Ritz values are accurate to $u\|A\|$ as long
as orthogonality holds to $\sqrt{u}$, because the error in the Ritz value goes like the
**square** of the orthogonality loss. Spending more is spending on nothing.

**The cost grows with $m$ because more Ritz values converge**, so more vectors need
orthogonalizing against. At $m = 20$ only one has converged and the cost is 12 operations. By
$m = 120$ many have, and the saving over full reorthogonalization narrows from 20 times to 5.
Selective wins most where it is most needed: long runs on matrices with a few well separated
eigenvalues.

### 3.2 Implicitly restarted Arnoldi

Keep a basis of $m = k + p$ vectors, where $k$ eigenvalues are wanted. When the basis is full,
apply $p$ shifted QR sweeps with the **unwanted** Ritz values as shifts, which filters them out
of the basis, then compress to $k$ vectors and extend back to $m$.

The essential step is that the Arnoldi factorization must be **continued**, not restarted:

```python
def arnoldi_extend(A, V, H, f, start, stop):
    """Extend A V = V H + f e^T from `start` to `stop` columns. All bounds come from the
    arguments, so any basis size works."""
    for j in range(start, stop):
        beta = np.linalg.norm(f)
        if beta < 1e-14 * max(1.0, np.linalg.norm(A)):
            return V, H, f, j                      # invariant subspace, stop early
        H[j, j - 1] = beta
        V[:, j] = f / beta
        w = A @ V[:, j]
        h = V[:, :j + 1].T @ w
        f = w - V[:, :j + 1] @ h
        h2 = V[:, :j + 1].T @ f                    # one reorthogonalization pass
        f = f - V[:, :j + 1] @ h2
        H[:j + 1, j] = h + h2
    return V, H, f, stop
```

and the restart itself is

```python
        shifts = np.real(theta[order[k:]])         # the UNWANTED Ritz values
        Qacc, Hk = np.eye(m), H[:m, :m].copy()
        for mu in shifts:
            Qq, Rr = np.linalg.qr(Hk - mu * np.eye(m))
            Hk = Rr @ Qq + mu * np.eye(m)
            Qacc = Qacc @ Qq
        Vn = V[:, :m] @ Qacc
        f = Vn[:, k] * Hk[k, k - 1] + f * Qacc[m - 1, k - 1]
        V[:, :k], H[:k, :k] = Vn[:, :k], Hk[:k, :k]
        V, H, f, _ = arnoldi_extend(A, V, H, f, k, m)
```

**Measured**, $k = 4$, $p = 8$, wanting the four largest of a spectrum
$\{25,24,23,22\} \cup [1,20]$, so the wanted set sits close to the bulk and one pass cannot
resolve it:

| $n$ | restarted, basis 12 | passes | one pass $m=12$ | one pass $m=48$ | one pass $m=96$ |
|---|---|---|---|---|---|
| 300 | $1.8\times10^{-15}$ | 9 | $1.6\times10^{-1}$ | $5.8\times10^{-15}$ | $6.3\times10^{-15}$ |
| 600 | $2.8\times10^{-15}$ | 9 | $1.5\times10^{-1}$ | $3.6\times10^{-15}$ | $5.0\times10^{-15}$ |
| 1000 | $2.5\times10^{-15}$ | 9 | $1.5\times10^{-1}$ | $4.5\times10^{-15}$ | $8.2\times10^{-15}$ |

**Twelve vectors reach machine precision where twelve vectors in one pass reach one digit.** The
restarted run matches what a single pass needs 48 vectors for, using a quarter of the storage:
96 kB against 384 kB at $n = 1000$. That is the whole reason ARPACK exists, and the reason its
interface asks for $k$ and $m$ rather than a tolerance.

**The number of passes does not grow with $n$** (9 at every size), because it is set by how hard
the spectrum is, not by the dimension.

**Restarting from a single vector does not work, and the failure is instructive.** An earlier
version restarted from $V\mathbf{q}_1$, one vector rather than the compressed $k$-dimensional
basis. It reached $10^{-15}$ after 5 restarts and then got **worse**, ending at $10^{-1}$ after
60. Repeatedly applying the filter polynomial drives the single vector toward the dominant
eigenvector and destroys the other $k-1$ directions. Keeping $k$ vectors is not an optimisation,
it is what makes the method work.

### 3.3 Block Lanczos and multiplicity

Start from $b$ vectors at once and orthogonalize blocks rather than vectors:

```python
def block_lanczos(A, V, m):
    """Block Lanczos from a block of b starting vectors. b comes from V.shape[1] and the size
    from V.shape[0], so any block size at any dimension works."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    V = np.asarray(V, dtype=float).reshape(n, -1)
    Q, _ = np.linalg.qr(V)
    blocks = [Q]
    for j in range(m):
        W = A @ blocks[-1]
        allQ = np.hstack(blocks)
        W = W - allQ @ (allQ.T @ W)
        W = W - allQ @ (allQ.T @ W)                # a second pass, for stability
        Qn, Bj = np.linalg.qr(W)
        keep = np.abs(np.diag(Bj)) > 1e-12 * np.linalg.norm(A)
        if not np.any(keep):                       # the block deflates, which is normal
            break
        blocks.append(Qn[:, keep])
        if sum(bl.shape[1] for bl in blocks) >= n:
            break
    QQ = np.hstack(blocks)
    return np.linalg.eigvalsh(QQ.T @ A @ QQ), QQ.shape[1]
```

**Counting correctly is most of the exercise.** Simply counting Ritz values near the target
confuses ghosts with genuine copies. The honest measure is the numerical **rank** of the block
of converged Ritz vectors:

```python
def eigenspace_rank(A, Q, target, tol=1e-6, res_tol=1e-6):
    """How many INDEPENDENT converged directions the basis Q holds in the target eigenspace."""
    T = Q.T @ A @ Q
    theta, S = np.linalg.eigh((T + T.T) / 2)
    pick = np.flatnonzero(np.abs(theta - target) < tol)
    if pick.size == 0:
        return 0
    Y = Q @ S[:, pick]
    res = np.linalg.norm(A @ Y - Y * theta[pick], axis=0)
    good = pick[res < res_tol * max(1.0, np.linalg.norm(A))]
    return 0 if good.size == 0 else int(np.linalg.matrix_rank(Q @ S[:, good], tol=1e-7))
```

**Measured**, eigenvalue 25 with the stated multiplicity, bulk in $[1,10]$, with a budget of
about 40 basis vectors:

| $n$ | true multiplicity | block size $b$ | vectors used | independent directions found |
|---|---|---|---|---|
| 100 | 3 | 1 | 41 | 2 |
| 100 | 3 | 2 | 42 | 2 |
| 100 | 3 | **3** | 42 | **3** |
| 150 | 4 | 1 | 41 | 2 |
| 150 | 4 | 2 | 42 | 2 |
| 150 | 4 | **4** | 44 | **4** |
| 200 | 5 | 1 | 41 | 2 |
| 200 | 5 | 2 | 42 | 2 |
| 200 | 5 | **5** | 45 | **5** |

**The block size caps what can be resolved.** With $b = 1$ or $b = 2$ the method finds 2
directions no matter how high the true multiplicity is; with $b$ equal to the multiplicity it
finds exactly the right number. That is the theoretical prediction of exercise 2.4 applied to a
block: a block of $b$ random vectors has a $\min(b, \text{mult})$-dimensional projection onto the
eigenspace, and the block Krylov space inherits it.

**Single-vector Lanczos finding 2 rather than 1 is roundoff, not information.** Exercise 2.4 says
it should find exactly 1. The second copy appears because roundoff introduces a component along
a second eigenvector, and once the first has converged the second grows. It is a ghost that
happens to be in the right place.

**And with a large enough budget even $b = 1$ recovers everything.** Repeating the same
measurement with about 70 basis vectors instead of 40, single-vector Lanczos finds all 3, 4 and
5 copies. Roundoff eventually supplies every direction. So the honest summary is that block
Lanczos resolves multiplicity **reliably and quickly**, while single-vector Lanczos may get there
eventually by accident, which is not something to depend on.

**The other reason blocks are used** is performance, not multiplicity: $A$ applied to $b$ vectors
at once is a BLAS-3 operation with arithmetic intensity $O(b)$, where $b$ separate matvecs are
BLAS-2. That is lesson 08's roofline again, and on modern hardware it is often the deciding
factor.

### 4.1 Where orthogonality is lost, against the gap

Measured, $n = 100$, spectrum $\{1+\text{gap}\} \cup [0,1]$, so a single eigenvalue sits above a
dense bulk and the gap is a free parameter:

| gap | $m$ at which loss exceeds $10^{-8}$ | $m$ at which loss exceeds $0.5$ |
|---|---|---|
| 0.01 | 61 | never within $m \le 100$ |
| 0.05 | 44 | never |
| 0.20 | 26 | 78 |
| 1.00 | 16 | 26 |
| 5.00 | 8 | 14 |
| 25.00 | **6** | **10** |

**A larger gap means earlier loss, and the effect is large**: a factor of 2500 in the gap moves
the onset from step 61 to step 6.

**The mechanism is Paige's theorem, and the causal chain runs the opposite way to intuition.** A
large gap means large $\gamma$ in the Kaniel-Paige bound (exercise 2.5), so the Ritz value
converges quickly. Paige's theorem says orthogonality is lost in the direction of **converged**
Ritz vectors, in proportion to $1/(\text{Ritz residual})$. Fast convergence therefore causes
early loss. Success is what breaks the method.

**And that is why the difficulty is unavoidable rather than a defect of the implementation.**
Nothing can be done about it by computing more carefully. The only remedies are to detect
convergence and reorthogonalize against what has converged (exercise 3.1), or to restart before
it happens (exercise 3.2). Both are responses to convergence, not to error.

### 4.2 Ghosts against $m$

Measured, $n = 60$, counting duplicated Ritz values (ghosts, since no eigenvalue is repeated)
and converged Ritz values, with no reorthogonalization:

| spectrum | $m=10$ | $m=20$ | $m=30$ | $m=40$ | $m=60$ | $m=90$ | $m=120$ |
|---|---|---|---|---|---|---|---|
| uniform $[1,10]$ | 0g / 0c | 0g / 0c | 0g / 3c | 0g / 7c | 0g / 42c | 14g / 60c | 36g / 60c |
| one big gap | 1g / 1c | 2g / 1c | 3g / 3c | 5g / 6c | 8g / 22c | 23g / 60c | 40g / 60c |
| geometric $[1,10^4]$ | 0g / 1c | 0g / 8c | 2g / 15c | 5g / 18c | 18g / 26c | 38g / 34c | 61g / 38c |

**Yes, the ghost count tracks convergence, and the "one big gap" row is the clearest case.** It
produces a ghost at $m = 10$, when exactly one Ritz value has converged, and thereafter the ghost
count grows roughly one per 10 steps as that value is rediscovered again and again.

**The uniform row is the control.** It produces **zero** ghosts all the way to $m = 60$, at which
point 42 Ritz values have converged and $m$ has reached $n$. Nothing converges early because
nothing is well separated, so nothing is lost early. The contrast between rows 1 and 2 at
$m = 10$ (0 ghosts against 1) is exactly the gap effect of exercise 4.1.

**Past $m = n$ the count grows roughly linearly**, since every further step can only re-find what
is already there. At $m = 120 = 2n$ the uniform spectrum has 36 ghosts among 120 Ritz values, so
about 30 percent of the output is spurious.

**The geometric row shows the third pattern.** Convergence saturates at 38 of 60 rather than
reaching all of them, because the small eigenvalues at the bottom of $[1,10^4]$ have negligible
relative gaps and never converge, while ghosts of the large ones accumulate. So a high ghost
count does **not** mean the run is nearly complete.

### 4.3 Naive, Arnoldi, and QR of the naive basis

Measured, $n = 60$, spectrum uniform in $[1,20]$:

| $m$ | $\kappa$(naive) | orth. loss, Arnoldi | orth. loss, QR of naive | subspace gap between them |
|---|---|---|---|---|
| 2 | $2.7\times10^{1}$ | $3.3\times10^{-16}$ | $4.4\times10^{-16}$ | below resolution |
| 6 | $1.8\times10^{7}$ | $4.4\times10^{-16}$ | $6.7\times10^{-16}$ | below resolution |
| 10 | $2.9\times10^{13}$ | $4.4\times10^{-16}$ | $6.7\times10^{-16}$ | below resolution |
| 12 | $3.3\times10^{16}$ | $4.4\times10^{-16}$ | $6.7\times10^{-16}$ | below resolution |
| 16 | $7.3\times10^{23}$ | $4.4\times10^{-16}$ | $6.7\times10^{-16}$ | $1.2\times10^{-5}$ |
| 20 | $1.3\times10^{30}$ | $4.4\times10^{-16}$ | $6.7\times10^{-16}$ | $3.7\times10^{-2}$ |
| 30 | $1.7\times10^{41}$ | $4.4\times10^{-16}$ | $6.7\times10^{-16}$ | **1.00** |

("Below resolution" means the measured gap sat at $1.5\times10^{-8} \approx \sqrt{u}$, which is
the accuracy of $\sqrt{1-\sigma_{\min}^2}$ when $\sigma_{\min}$ is near 1, so no smaller gap can
be resolved this way.)

**Both bases are orthonormal to machine precision, at every $m$, and that proves nothing.** The
QR of the naive basis is a perfectly orthonormal set of vectors. The question is whether they
span the right space.

**They do, up to about $m = 12$, and then they do not.** The gap is unresolvable through
$m = 12$, then $1.2\times10^{-5}$ at $m = 16$, $3.7\times10^{-2}$ at $m = 20$, and **1.00** at
$m = 30$, meaning the two subspaces are completely different: one contains a direction orthogonal
to the whole of the other.

**The crossover is exactly where $\kappa$(naive) reaches $1/u$.** At $m = 12$, $\kappa =
3.3\times10^{16}$ and $1/u = 9\times10^{15}$. Beyond that the stored naive vectors are
numerically rank deficient, so QR is factoring a matrix whose columns no longer determine the
intended subspace. QR is backward stable and returns an exact factorization of a matrix within
roundoff of the one given, but the one given has already lost the information.

**This is a clean instance of lesson 06's separation.** The QR algorithm is stable; the problem
of recovering $K_m$ from the stored naive basis is ill conditioned; and a stable algorithm on an
ill conditioned problem still gives a wrong answer. Arnoldi avoids it not by computing better but
by **never forming the ill conditioned basis**, orthogonalizing each vector as it is produced,
while the information is still there.

### 5.1 Paige's theorem, and testing the proportionality

**The statement.** For Lanczos in floating point, let $(\theta_k,\mathbf{y}_k)$ be a Ritz pair at
step $m$, with residual bound $|\beta_m s_{mk}|$. Then

$$|\mathbf{q}_{m+1}^T\mathbf{y}_k| \approx \frac{u\|A\|}{|\beta_m s_{mk}|}.$$

Two claims are packed in: the loss is in the **direction** of converged Ritz vectors, and it is
**inversely proportional** to the Ritz residual.

**Testing the direction alone is not a test.** With a matrix that has one dominant eigenvalue,
almost any vector correlates with the top Ritz vector, so a positive result means little. The
proportionality is the falsifiable claim, and the way to test it is to measure the **product**
$|\beta_m s_{mk}| \cdot |\mathbf{q}_{m+1}^T\mathbf{y}_k|$ and check that it stays constant while
each factor moves over many orders of magnitude.

**Measured**, $n = 100$, spectrum $\{30\} \cup [1,10]$, no reorthogonalization, tracking the
top Ritz pair:

| $m$ | Ritz residual $\lvert\beta_m s_{mk}\rvert$ | overlap $\lvert\mathbf{q}_{m+1}^T\mathbf{y}_k\rvert$ | **product** | orth. loss |
|---|---|---|---|---|
| 4 | $1.49\times10^{-1}$ | $9.35\times10^{-15}$ | $1.39\times10^{-15}$ | $7.2\times10^{-15}$ |
| 7 | $1.01\times10^{-4}$ | $1.13\times10^{-11}$ | $1.14\times10^{-15}$ | $9.7\times10^{-12}$ |
| 10 | $7.78\times10^{-8}$ | $1.61\times10^{-8}$ | $1.26\times10^{-15}$ | $1.4\times10^{-8}$ |
| 13 | $5.54\times10^{-11}$ | $1.77\times10^{-5}$ | $9.84\times10^{-16}$ | $1.5\times10^{-5}$ |
| 16 | $4.84\times10^{-14}$ | $2.37\times10^{-2}$ | $1.15\times10^{-15}$ | $2.0\times10^{-2}$ |
| 19 | $2.26\times10^{-15}$ | $3.53\times10^{-1}$ | $7.99\times10^{-16}$ | $7.8\times10^{-1}$ |

**The residual falls through fourteen orders of magnitude, the overlap rises through fourteen,
and the product stays within a factor of 1.7 of $1.1\times10^{-15}$.** That is the
proportionality, confirmed quantitatively rather than in direction only.

**The constant is the right size too.** $u\|A\| = 1.1\times10^{-16} \times 30 =
3.3\times10^{-15}$, and the measured product is $1.1\times10^{-15}$, one third of it. Paige's
result is stated up to a modest constant, so agreement to a factor of 3 is agreement.

**Beyond $m = 19$ the relation stops holding**, and it should. Orthogonality has reached 0.78 by
then, so $\mathbf{q}_{m+1}$ is no longer orthogonal to anything, the Ritz vector is no longer a
meaningful approximation, and the theory's hypotheses are gone. The valid range is exactly the
range in which orthogonality is being lost, which is the range the theory is about.

**The design lesson.** A theory that predicts a proportionality should be tested by holding the
product fixed, not by confirming the sign of an effect. Confirming the direction here would have
been trivially easy and would have distinguished nothing.

### 5.2 Why Krylov spaces are optimal

**Claim.** Among all methods that access $A$ only through $m$ matrix-vector products started
from $\mathbf{v}$, none can produce an approximation better than the best element of
$K_m(A,\mathbf{v})$.

**Proof.** By exercise 1.1, the set of vectors constructible after $m$ products is exactly
$K_m$. An approximation produced by any such method is therefore an element of $K_m$, so it is
no better than the best element of $K_m$ measured in whatever norm. There is nothing more to
prove; the content is entirely in exercise 1.1's induction.

**What the claim does and does not say.**

**It is a statement about information, not about arithmetic.** The method may do unlimited work
between products: solve dense subproblems, run an eigensolver on $H_m$, take any combination it
likes. None of that enlarges the reachable set.

**It says the search space is optimal, not that any given method finds the best point in it.**
CG does find it, in the $A$-norm (exercise 24.5.1). GMRES finds it in the residual 2-norm.
Arnoldi's Ritz values are **not** optimal in general: for a nonnormal matrix the best
approximate eigenvector in $K_m$ can be much better than the Ritz vector, which is exercise 5.3.

**It assumes the products all start from one vector.** Block methods (exercise 3.3) use $b$
starting vectors, and their reachable set is the block Krylov space, which is strictly larger.
The optimality statement then applies to that space instead, and the cost is $b$ products per
step. So the theorem does not say blocks are pointless, it says the accounting must be per
product.

**And the assumption can fail entirely.** A method with access to $A^T\mathbf{v}$ reaches a
larger space, which is what BiCG uses. A method that can solve with $A - \sigma I$ reaches a
rational Krylov space, which is shift and invert. Both escape the theorem by having more
information, not by being cleverer, and that is the honest way to read every optimality result:
it fixes the information and asks what can be done with it.

### 5.3 Non-normality

**The problem.** For a nonnormal $A$ the eigenvectors are not orthogonal and can be nearly
parallel. Three things then go wrong at once.

**Ritz values can be poor even when $K_m$ is rich.** The Rayleigh quotient $H_m = Q_m^TAQ_m$ is
an orthogonal projection, which is the right thing for a symmetric matrix because its
eigenvectors are orthogonal. For a nonnormal matrix the natural projection is **oblique**, along
the left eigenvectors, and lesson 16 measured that an oblique projector has
$\|P\|_2 = 1/\cos\theta_{\max}$, which is large when left and right eigenvectors are nearly
orthogonal to each other. The orthogonal projection then misses badly.

**The residual is not a bound on the error.** For symmetric $A$ a small residual
$\|A\mathbf{y}-\theta\mathbf{y}\|$ bounds the eigenvalue error by the same amount. For nonnormal
$A$ the bound acquires the eigenvector condition number $1/|\mathbf{w}^T\mathbf{v}|$, which can
be enormous, so a small Arnoldi residual does not mean a small error. Lesson 06's separation of
problem from algorithm, again.

**Eigenvalues stop describing behaviour.** This is what pseudospectra (lesson 40) exist for. The
$\varepsilon$-pseudospectrum

$$\Lambda_\varepsilon(A) = \{z : \|(zI-A)^{-1}\| > 1/\varepsilon\}$$

is a small neighbourhood of the eigenvalues when $A$ is normal, and can be enormous when it is
not. Where it extends tells you about transient behaviour that the eigenvalues do not: exercise
23.4.2 measured an iteration with $\rho(G) = 0.89$ whose error grew by nineteen orders of
magnitude first, and that growth is exactly what a pseudospectrum extending outside the unit
disc predicts.

**A route through the investigation.** Take a family interpolating between normal and highly
nonnormal, for example $A(t) = D + tN$ with $D$ diagonal and $N$ strictly upper triangular, and
measure at each $t$: the Ritz value error against the Arnoldi residual, the ratio between them,
and the extent of the pseudospectrum. The first two agree at $t = 0$ and separate as $t$ grows,
by a factor that tracks the pseudospectral radius.

**The connection to lesson 27 is the deepest part of the exercise.** Greenbaum, Ptak and
Strakos showed that **any** decreasing residual curve can be produced by GMRES on a matrix with
**any** prescribed eigenvalues. So the eigenvalues of a nonsymmetric matrix carry **no
information at all** about how fast GMRES converges on it. That is a much stronger statement than
"the bound is loose": there is no bound in terms of eigenvalues to be had. What does govern
convergence is the pseudospectrum, or equivalently the field of values, which are the
non-normality-aware substitutes for the spectrum.

**And this is why lesson 26 spends its effort on the symmetric case.** Everything comfortable
about Lanczos, the short recurrence, the real Ritz values interlacing the spectrum, the residual
bounding the error, comes from symmetry. Dropping it does not merely cost storage, it removes
the theory.

---

## Lesson 27, GMRES and Nonsymmetric Solvers

### 1.1 Why GMRES's residual cannot increase and BiCG's can

GMRES **minimises** $\|\mathbf{b} - A\mathbf{x}\|_2$ over $\mathbf{x}_0 + K_m$, and
$K_m \subseteq K_{m+1}$. A minimum over a larger set cannot be larger than a minimum over a
smaller one, so

$$\|\mathbf{r}_{m+1}\| = \min_{\mathbf{x} \in \mathbf{x}_0+K_{m+1}}\|\mathbf{b}-A\mathbf{x}\|
\le \min_{\mathbf{x} \in \mathbf{x}_0+K_m}\|\mathbf{b}-A\mathbf{x}\| = \|\mathbf{r}_m\|.$$

That is the whole proof, and notice it uses nothing about $A$: not symmetry, not
definiteness, not even invertibility.

**BiCG minimises nothing.** Its iterate is determined by a **biorthogonality** condition,
$\mathbf{r}_m \perp \mathbf{s}_0,\dots,\mathbf{s}_{m-1}$, which fixes $\mathbf{x}_m$ uniquely
but says nothing about the size of the residual. A Galerkin condition of that kind is a
projection, and a projection can be large when the projector is oblique, which lesson 16
measured as $\|P\|_2 = 1/\cos\theta_{\max}$. Measured in lesson 27 section 8: BiCG's residual
peaked at **92 times** its starting value before converging.

**The practical consequence is the stopping rule.** GMRES's residual is monotone and free, so
the first time it drops below the tolerance you are done. BiCG's can dip below the tolerance
and come back up, so a run stopped on the first crossing may be stopped at a fluke.

### 1.2 GMRES(30) has stalled after 300 steps

**Two things to try, and the second is much more likely to help.**

**Increase the restart parameter.** Lesson 27 section 4 measured GMRES(20) at 7426 steps where
GMRES(50) took 2660 and full GMRES took 100, and a smaller $k$ can stagnate permanently. If 30
vectors fit, 60 probably do too, and the gain is often enormous.

**Precondition.** This is the one that matters. Restarting stalls because the useful progress
does not happen within a cycle, and a preconditioner changes what "useful progress" means by
clustering the spectrum, or here by pulling the field of values away from the origin. Lesson 25
section 4 measured preconditioners changing the exponent, not just the constant, which no
choice of $k$ can do.

**Why preconditioning is more likely.** Raising $k$ buys a bounded improvement: full GMRES is
the ceiling, and if full GMRES also converges slowly then no $k$ helps. Preconditioning changes
the problem, so it has no such ceiling. And a stall at 300 steps with $k = 30$ means ten cycles
have made no progress, which suggests the difficulty is not the restart length.

**A third option, if the stagnation is caused by a few small eigenvalues**, is deflated
restarting, and exercise 3.3 measures it turning "never converges" into 110 steps.

### 1.3 Tightly clustered eigenvalues near 1

**The reasoning is wrong because it imports a symmetric fact into a nonsymmetric setting.**

For a **symmetric** matrix, lesson 24 section 6 established that clustering decides everything:
with $m$ distinct eigenvalues CG finishes in $m$ steps whatever $\kappa$ is. The proof puts a
polynomial root at each cluster, and it works because a symmetric matrix has an **orthogonal**
eigenvector basis, so $\|q(A)\| = \max_i|q(\lambda_i)|$.

For a nonsymmetric matrix that identity fails. The right statement involves the eigenvector
matrix:

$$\|q(A)\| \le \kappa(V)\max_i|q(\lambda_i)|,$$

and $\kappa(V)$ can be astronomically large. Section 5 measured $\kappa(V) = 3\times10^{23}$ on
a matrix whose eigenvalues were the integers 1 to 40.

**Theorem 27.4 makes it worse than "the bound is loose".** Greenbaum, Ptak and Strakos showed
that **any** non-increasing residual curve can be produced with **any** prescribed eigenvalues.
So there is no bound in terms of the eigenvalues to be had, loose or otherwise, and a spectrum
clustered at 1 is compatible with GMRES stagnating for as long as you like.

**What to ask for instead.** Is the matrix close to normal? If $\kappa(V)$ is modest the
eigenvalue reasoning is fine. If not, ask for the **field of values**: if the origin is well
outside it, Elman's bound applies, and section 6 measured that it is never violated.

### 2.1 Proof of Proposition 27.1

The iterate at step $m$ is $\mathbf{x} = \mathbf{x}_0 + Q_m\mathbf{y}$ for
$\mathbf{y} \in \mathbb{R}^m$, since $Q_m$'s columns are a basis for $K_m$. Then

$$\mathbf{b} - A\mathbf{x} = \mathbf{r}_0 - AQ_m\mathbf{y}
= \mathbf{r}_0 - Q_{m+1}\tilde{H}_m\mathbf{y}$$

using the Arnoldi relation. The first Arnoldi vector is $\mathbf{q}_1 = \mathbf{r}_0/\beta$
with $\beta = \|\mathbf{r}_0\|$, and $\mathbf{q}_1$ is the first column of $Q_{m+1}$, so
$\mathbf{r}_0 = \beta Q_{m+1}\mathbf{e}_1$. Therefore

$$\mathbf{b} - A\mathbf{x} = Q_{m+1}\left(\beta\mathbf{e}_1 - \tilde{H}_m\mathbf{y}\right).$$

**Now the step that needs orthonormality.** $Q_{m+1}$ has orthonormal columns, so
$Q_{m+1}^TQ_{m+1} = I_{m+1}$ and for any $\mathbf{z} \in \mathbb{R}^{m+1}$,

$$\|Q_{m+1}\mathbf{z}\|_2^2 = \mathbf{z}^TQ_{m+1}^TQ_{m+1}\mathbf{z} = \|\mathbf{z}\|_2^2.$$

Note that $Q_{m+1}$ is **not** square, so it is not an orthogonal matrix and
$Q_{m+1}Q_{m+1}^T \ne I$. What is true, and all that is needed, is that it is an **isometry**
from $\mathbb{R}^{m+1}$ into $\mathbb{R}^n$. Hence

$$\|\mathbf{b} - A\mathbf{x}\|_2 = \|\beta\mathbf{e}_1 - \tilde{H}_m\mathbf{y}\|_2,$$

and minimising the left side over $\mathbf{x} \in \mathbf{x}_0+K_m$ is the same as minimising
the right side over $\mathbf{y} \in \mathbb{R}^m$.

**And the 2-norm is not a choice here.** The argument works because $Q$ preserves that norm
and no other. Minimising the residual in a different norm would not reduce to a small least
squares problem, which is why GMRES minimises the 2-norm and why CG, which minimises the
$A$-norm, needs a different derivation.

### 2.2 One rotation suffices per step

Suppose after step $j-1$ the rotations $G_1,\dots,G_{j-1}$ have reduced
$\tilde{H}_{j-1}$ to upper triangular. Column $j$ of $\tilde{H}_j$ is

$$\mathbf{h} = (h_{1j},\dots,h_{jj},h_{j+1,j},0,\dots,0)^T,$$

with $h_{j+1,j}$ in position $j+1$ and zeros below, because $\tilde{H}$ is Hessenberg.

Rotation $G_i$ acts on rows $i$ and $i+1$ only. Applying $G_1,\dots,G_{j-1}$ mixes rows 1
through $j$ and **leaves row $j+1$ untouched**, since the last of them acts on rows $j-1$ and
$j$. So after the replay the column is still

$$(\tilde{h}_{1j},\dots,\tilde{h}_{jj},h_{j+1,j},0,\dots,0)^T,$$

with exactly one nonzero below the diagonal, namely $h_{j+1,j}$ in position $j+1$. One rotation
$G_j$ acting on rows $j$ and $j+1$ removes it.

**The cost accounting follows immediately.** The replay is $j-1$ rotations at $O(1)$ each and
the new rotation is $O(1)$, so the least squares update is $O(j)$, not $O(j^3)$. Over $m$ steps
that is $O(m^2)$ scalar work, negligible beside the $O(m^2n)$ of orthogonalization.

**And rows below $j+1$ never contain anything**, which is exactly why the Hessenberg structure
is what makes GMRES affordable. For a full matrix each column would have $n$ entries below the
diagonal and the update would be $O(n)$ rotations.

### 2.3 GMRES terminates at the degree of the minimal polynomial

Let $d$ be the degree of the minimal polynomial of $A$ with respect to $\mathbf{r}_0$, that is
the smallest $d$ with $p(A)\mathbf{r}_0 = \mathbf{0}$ for some monic $p$ of degree $d$.

**Termination.** By exercise 26.2.4's argument, $\dim K_m = \min(m, d)$, so $K_d = K_{d+1}$ and
$K_d$ is invariant under $A$. Since $A$ is invertible and $\mathbf{r}_0 \in K_d$, we have
$A^{-1}\mathbf{r}_0 \in K_d$, so the exact correction lies in the space GMRES is searching and
the minimum is zero.

Explicitly: write $p(t) = t^d + c_{d-1}t^{d-1} + \dots + c_0$ with $p(A)\mathbf{r}_0 = 0$. Then
$c_0 \ne 0$, since otherwise $t$ divides $p$ and $A^{-1}$ applied to
$p(A)\mathbf{r}_0 = 0$ would give a monic polynomial of degree $d-1$ annihilating
$\mathbf{r}_0$, contradicting minimality. Solving for the constant term,

$$A^{-1}\mathbf{r}_0 = -\frac{1}{c_0}\left(A^{d-1} + c_{d-1}A^{d-2} + \dots + c_1I\right)
\mathbf{r}_0 \in K_d.$$

**Why this is the right generalisation.** For a diagonalizable $A$, the minimal polynomial with
respect to $\mathbf{r}_0$ has one root per **distinct** eigenvalue present in $\mathbf{r}_0$'s
expansion, so $d$ is the number of distinct eigenvalues represented, which is lesson 24's
statement exactly.

**But it covers two cases lesson 24 did not.** A defective matrix has a minimal polynomial with
repeated roots, so $d$ can exceed the number of distinct eigenvalues: the convection diffusion
operator at $pe = 2$ is a single Jordan block, its minimal polynomial has degree $n$, and
section 5's measurement shows GMRES taking exactly $n$ steps for every $pe > 0$. And a
right-hand side with no component in some eigenvector makes $d$ smaller, which is exercise
26.2.4's condition $c_i \ne 0$.

**A caution.** $d$ bounds the count and rarely describes it. For the cyclic shift $d = n$ and
GMRES really does need all $n$ steps; for a matrix with $d = n$ but a favourable spectrum it may
finish in ten. Finite termination is a fact about exact arithmetic and a small enough matrix,
not a convergence theory.

### 2.4 The field of values is convex

**Toeplitz-Hausdorff.** $W(A) = \{\mathbf{x}^\ast A\mathbf{x} : \|\mathbf{x}\| = 1\}$ is convex.

**Reduction to two dimensions.** Take $\alpha, \beta \in W(A)$ attained at unit vectors
$\mathbf{u}, \mathbf{v}$, which we may assume linearly independent (otherwise
$\alpha = \beta$). Let $S = \operatorname{span}(\mathbf{u},\mathbf{v})$ and let $B$ be the
$2\times2$ compression of $A$ to an orthonormal basis of $S$. Then $W(B) \subseteq W(A)$ and
$\alpha,\beta \in W(B)$, so it suffices to prove the $2\times2$ case.

**The $2\times2$ case.** Any $2\times2$ matrix can be written $B = \gamma I + \delta C$ with
$\gamma,\delta$ complex and $C$ having trace zero, and $W(\gamma I + \delta C) = \gamma +
\delta W(C)$, an affine image, so convexity for $C$ gives it for $B$. A trace-zero
$2\times2$ matrix is unitarily similar to $\begin{pmatrix}0 & a \\ b & 0\end{pmatrix}$, whose
field of values is an **ellipse** centred at the origin with axes $|a|+|b|$ and
$\big||a|-|b|\big|$. An ellipse is convex.

**Equality with the convex hull of the eigenvalues, for normal $A$.** If $A = U\Lambda U^\ast$
with $U$ unitary, substitute $\mathbf{y} = U^\ast\mathbf{x}$:

$$\mathbf{x}^\ast A\mathbf{x} = \mathbf{y}^\ast\Lambda\mathbf{y} = \sum_i \lambda_i|y_i|^2,$$

and $\sum_i|y_i|^2 = 1$ with $|y_i|^2 \ge 0$. So $W(A)$ is exactly the set of convex
combinations of the eigenvalues, that is their convex hull. Every convex combination is
attainable by choosing the $|y_i|^2$ freely.

**For nonnormal $A$ it is strictly larger**, and that gap is the point. Measured for the
bidiagonal family of section 5: the eigenvalues are $1,\dots,40$, so their convex hull is the
interval $[1,40]$ and the origin is at distance 1 from it. But at $\gamma = 3$ the field of
values **contains** the origin, so it is strictly larger than the hull.

### 2.5 Elman's bound

Let $\mathbf{r}_0$ be the initial residual and $\mathbf{r}_k$ the GMRES residual at step $k$.
Consider one step of a **Richardson-like** method with the optimal step: since GMRES minimises
over all polynomials, its residual is at least as small as the one from any particular choice.
Take the choice $\mathbf{r} \leftarrow \mathbf{r} - \alpha A\mathbf{r}$ with $\alpha$ optimal:

$$\min_\alpha\|\mathbf{r} - \alpha A\mathbf{r}\|^2
= \|\mathbf{r}\|^2 - \frac{(\mathbf{r}^TA\mathbf{r})^2}{\|A\mathbf{r}\|^2}
= \|\mathbf{r}\|^2\left(1 - \frac{(\mathbf{r}^TA\mathbf{r})^2}
{\|\mathbf{r}\|^2\|A\mathbf{r}\|^2}\right).$$

Now bound the two pieces. The denominator satisfies
$\|A\mathbf{r}\| \le \|A\|_2\|\mathbf{r}\|$. For the numerator,

$$\frac{\mathbf{r}^TA\mathbf{r}}{\|\mathbf{r}\|^2} \in W(A)$$

by definition, so **if the origin lies outside $W(A)$ at distance $\nu$**, then
$|\mathbf{r}^TA\mathbf{r}|/\|\mathbf{r}\|^2 \ge \nu$. Substituting,

$$\|\mathbf{r}_{k+1}\| \le \|\mathbf{r}_k\|\sqrt{1 - \frac{\nu^2}{\|A\|_2^2}},$$

and iterating gives the bound.

**Where the hypothesis is used, precisely.** It is the single step bounding the Rayleigh
quotient away from zero. If the origin is inside $W(A)$ then some unit vector has
$\mathbf{r}^TA\mathbf{r} = 0$, the whole gain term vanishes, and the bound degenerates to
$\|\mathbf{r}_{k+1}\| \le \|\mathbf{r}_k\|$, which is Proposition 27.2 and says nothing new.

**Measured, this is exactly what happens.** Section 6's last two rows have the origin inside
$W(A)$ and the bound reports $\rho = 1.00000$, giving no information, while GMRES converges in
29 and 25 steps. The bound is not wrong there, it is empty.

**And note what the derivation gives away.** It bounds GMRES by a **one step** method, so it
cannot see any benefit from the higher degree polynomials GMRES actually uses. That is why the
bound is loose by factors of 2 to 1800 in the measurements.

### 3.1 Householder GMRES against modified Gram-Schmidt

Modified Gram-Schmidt loses orthogonality at a rate proportional to the condition number of the
vectors being orthogonalized. Householder reflections do not: each is exactly orthogonal to
roundoff by construction, so the product of them is too.

```python
def arnoldi_householder(A, v, m):
    """Arnoldi via Householder reflections. Sizes come from A and v; m is clipped to n."""
    n = A.shape[0]
    m = min(m, n)
    W = np.zeros((n, m + 1))                    # the reflector vectors
    H = np.zeros((m + 1, m))
    Q = np.zeros((n, m + 1))
    z = np.asarray(v, dtype=float).ravel().copy()
    for j in range(m + 1):
        x = z[j:].copy()                        # reflect the tail onto the axis
        alpha = -np.sign(x[0] if x[0] != 0 else 1.0) * np.linalg.norm(x)
        u = x.copy()
        u[0] -= alpha
        nu = np.linalg.norm(u)
        W[j:, j] = u / nu if nu > 0 else 0.0
        z[j:] = 0.0
        z[j] = alpha
        if j > 0:
            H[:j + 1, j - 1] = z[:j + 1]
        q = np.zeros(n)                         # q_j = P_0 P_1 ... P_j e_j
        q[j] = 1.0
        for k in range(j, -1, -1):
            q -= 2.0 * W[:, k] * (W[:, k] @ q)
        Q[:, j] = q
        if j == m:
            break
        z = A @ q                               # z = P_j ... P_0 A q_j
        for k in range(j + 1):
            z -= 2.0 * W[:, k] * (W[:, k] @ z)
    return Q, H[:m + 1, :m]
```

**Measured**, $\max_{i \ne j}|\mathbf{q}_i^T\mathbf{q}_j|$ over the basis, $n = 60$:

| matrix | $m$ | MGS | MGS twice | Householder |
|---|---|---|---|---|
| well conditioned | 10 | $4.4\times10^{-16}$ | $4.4\times10^{-16}$ | $8.9\times10^{-16}$ |
| | 30 | $7.7\times10^{-16}$ | $4.4\times10^{-16}$ | $1.1\times10^{-15}$ |
| | 55 | $2.8\times10^{-13}$ | $5.6\times10^{-16}$ | $1.2\times10^{-15}$ |
| nonnormal, $\gamma = 30$ | 10 | $4.4\times10^{-16}$ | $6.7\times10^{-16}$ | $1.2\times10^{-15}$ |
| | 30 | $1.1\times10^{-14}$ | $6.7\times10^{-16}$ | $1.2\times10^{-15}$ |
| | 55 | **$2.8\times10^{-4}$** | $6.7\times10^{-16}$ | $1.2\times10^{-15}$ |
| ill conditioned, $\kappa = 10^{10}$ | 30 | $9.9\times10^{-12}$ | $6.7\times10^{-16}$ | $5.6\times10^{-16}$ |
| | 55 | **$6.6\times10^{-8}$** | $6.7\times10^{-16}$ | $1.3\times10^{-15}$ |

**Plain MGS loses four digits of orthogonality by $m = 55$ on the nonnormal matrix**, and eight
on the ill conditioned one. Householder holds $10^{-15}$ throughout.

**But one reorthogonalization pass does just as well**, and that is the practically important
row: MGS twice matches Householder everywhere, at $6.7\times10^{-16}$.

**Cost**, $n = 200$, $m = 150$:

| method | time |
|---|---|
| MGS | 0.043 s |
| MGS with one reorthogonalization pass | 0.083 s |
| Householder | 0.097 s |

**So the reorthogonalized MGS is slightly cheaper and much simpler**, which is why `nalib`'s
GMRES does that rather than Householder. Householder's advantage is a **guarantee** rather than
a heuristic: its orthogonality is bounded a priori while "twice is enough" is a rule of thumb,
justified by Kahan and Parlett's analysis but with hypotheses. In a library that must never
surprise anyone, the guarantee is worth the extra line count.

### 3.2 Flexible GMRES

Plain GMRES applies $M^{-1}$ and then orthogonalizes, storing only the Arnoldi basis $Q$. That
is enough because $Q_m$ and one fixed $M^{-1}$ determine the iterate. If $M$ **changes** from
step to step there is no single operator whose Krylov space is being built, and the Arnoldi
relation $M^{-1}AQ_m = Q_{m+1}\tilde{H}_m$ is simply false.

**Flexible GMRES fixes it by storing the preconditioned vectors too**, and correcting with
those:

```python
def fgmres(A, b, M, restart, max_iter, tol=1e-10):
    """GMRES that stores the PRECONDITIONED basis, so M may change from step to step."""
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    m = min(restart, n)
    x = np.zeros(n)
    scale = float(np.linalg.norm(b)) or 1.0
    total = 0
    while total < max_iter:
        r = b - A @ x
        beta = float(np.linalg.norm(r))
        if beta <= tol * scale:
            break
        Q = np.zeros((n, m + 1))
        Z = np.zeros((n, m))                 # the preconditioned vectors, kept as well
        H = np.zeros((m + 1, m))
        Q[:, 0] = r / beta
        used = 0
        for j in range(m):
            Z[:, j] = M(Q[:, j], total + j)  # a DIFFERENT preconditioner each step
            w = A @ Z[:, j]
            for i in range(j + 1):
                H[i, j] = Q[:, i] @ w
                w = w - H[i, j] * Q[:, i]
            H[j + 1, j] = float(np.linalg.norm(w))
            used = j + 1
            if H[j + 1, j] < 1e-14:
                break
            Q[:, j + 1] = w / H[j + 1, j]
        rhs = np.zeros(used + 1)
        rhs[0] = beta
        y, *_ = np.linalg.lstsq(H[:used + 1, :used], rhs, rcond=None)
        x = x + Z[:, :used] @ y              # correct with Z, not Q: that is the whole trick
        total += used
    return x, total
```

**Measured**, convection diffusion at $m = 120$, $pe = 3$, tolerance $10^{-10}$, using an inner
GMRES of 3 to 5 steps as the preconditioner:

| run | steps | final true relative residual |
|---|---|---|
| plain GMRES(20), no preconditioner | 476 | converged |
| **flexible GMRES(20), varying inner solve** | **160** | $9.0\times10^{-13}$ |
| plain GMRES(20) fed the varying $M$ | 1039 | **$1.1\times10^{-5}$** |

**The third row is the point, and it is worse than being slow: it is wrong.** Plain GMRES
stopped after 1039 steps believing it had converged, because the quantity it monitors is the
residual of the least squares problem it thinks it is solving. With a varying $M$ that problem
is not the real one, so the reported residual and the actual residual part company, and the
actual one is $10^{-5}$.

**That is the failure to remember.** Feeding a varying preconditioner to plain GMRES does not
merely slow it down or make it diverge visibly. It makes the **stopping rule lie**, which is
the same class of error as lesson 06's small residual with a large forward error.

**Where a varying preconditioner comes from**, and why anyone would want one: an inner
iterative solve, a multigrid cycle with an adaptive number of sweeps, or a preconditioner
itself computed to a loose tolerance. All of them are cheaper than a fixed exact one, and all
of them require FGMRES.

### 3.3 GMRES with deflated restarting

Carry $k$ approximate eigenvectors across the restart boundary, so the information the restart
would have thrown away survives. **They must be harmonic Ritz vectors, not ordinary ones.**

```python
def gmres_dr(A, b, k, m, max_outer=500, tol=1e-10):
    """GMRES(m) carrying k HARMONIC Ritz vectors across each restart.

    Harmonic Ritz pairs solve (AV)^T A V y = theta (AV)^T V y. Unlike ordinary Ritz pairs they
    approximate the SMALLEST eigenvalues well, which are the ones costing the iterations.
    """
    import scipy.linalg as sla

    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    m = min(m, n)
    x = np.zeros(n)
    scale = float(np.linalg.norm(b)) or 1.0
    U = np.zeros((n, 0))                          # the deflation space, empty to begin
    total = 0
    for _ in range(max_outer):
        r = b - A @ x
        if np.linalg.norm(r) <= tol * scale:
            return total, True
        V = np.zeros((n, m))
        nc = U.shape[1]
        if nc:
            V[:, :nc], _ = np.linalg.qr(U)        # the carried vectors go in first
        v0 = r - (V[:, :nc] @ (V[:, :nc].T @ r) if nc else 0.0)
        V[:, nc] = v0 / np.linalg.norm(v0)
        cols = nc + 1
        for j in range(nc, m - 1):                # Arnoldi fills the rest
            w = A @ V[:, j]
            w -= V[:, :cols] @ (V[:, :cols].T @ w)
            w -= V[:, :cols] @ (V[:, :cols].T @ w)
            nw = float(np.linalg.norm(w))
            if nw < 1e-13:
                break
            V[:, cols] = w / nw
            cols += 1
        V = V[:, :cols]
        AV = A @ V
        y, *_ = np.linalg.lstsq(AV, r, rcond=None)
        x = x + V @ y
        total += cols
        vals, vecs = sla.eig(AV.T @ AV, AV.T @ V)  # the harmonic Ritz problem
        keep = np.isfinite(vals)
        vals, vecs = vals[keep], vecs[:, keep]
        U = np.real(V @ vecs[:, np.argsort(np.abs(vals))[:k]])
    return total, False
```

**Measured**, $n = 120$, spectrum $\{10^{-4},10^{-3},10^{-2}\} \cup [1,10]$ so three isolated
small eigenvalues carry all the difficulty, $\kappa = 10^5$. Full GMRES takes **67** steps:

| basis $m$ | plain | deflating 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| 10 | **never** | 3970 | 670 | **110** | 150 |
| 20 | 3175 | 180 | 120 | **100** | 100 |
| 30 | 1421 | 150 | 90 | **90** | 90 |

**Deflating exactly the three offenders is what it takes.** At basis 10, plain GMRES never
converges and deflating 3 turns it into 110 steps. At basis 20 it is 3175 against 100, a factor
of 32. Deflating more than 3 gains nothing, which is right: there are only three outliers.

**Harmonic Ritz vectors matter**, and using ordinary ones is the easy mistake:

| basis 20, deflating | harmonic | ordinary |
|---|---|---|
| 1 | 180 | 340 |
| 2 | 120 | 200 |
| 3 | 100 | 200 |

Ordinary Ritz values converge to the **largest** eigenvalues first (lesson 24 exercise 3.3
measured that), so taking the $k$ smallest of them gives poor approximations to the smallest
eigenvalues, which are the ones costing the iterations. Harmonic Ritz values are the ordinary
Ritz values of $A^{-1}$ in disguise, so they converge to the small end.

**It does not fix section 4's cyclic shift, and the reason is worth understanding.** Measured:
deflating 1, 3, 5 or 8 vectors leaves GMRES(10) stuck at residual 1.0 exactly as before. The
cyclic shift's eigenvectors are Fourier vectors with every component of modulus
$1/\sqrt{n}$, so there is no small invariant subspace to deflate; and the Krylov space is
$\operatorname{span}(\mathbf{e}_1,\dots,\mathbf{e}_m)$ while the answer is
$\mathbf{e}_n$, so **no subspace of dimension below $n$ contains any part of the solution**.
Deflation carries vectors **from** the Krylov space, so it cannot supply what that space lacks.

**Deflation cures stagnation caused by a few isolated eigenvalues. It does not cure stagnation
caused by the Krylov space itself being useless.** Distinguishing the two is the diagnostic
worth taking away.

### 4.1 The Peclet sweep

Measured, GMRES steps to a relative residual of $10^{-8}$ with $\mathbf{b} = \mathbf{1}$:

| $pe$ | $m=40$ | $m=80$ | $m=160$ | $\kappa(V)$ at $m=80$ |
|---|---|---|---|---|
| 0.0 | **20** | **40** | **80** | $1.000$ |
| 0.5 | 40 | 80 | 160 | $5.9\times10^{8}$ |
| 1.0 | 40 | 80 | 160 | $1.0\times10^{18}$ |
| 1.9 | 40 | 80 | 160 | $7.0\times10^{19}$ |
| **2.0** | 40 | 80 | 160 | **$\infty$** |
| 2.1 | 40 | 80 | 160 | $1.2\times10^{41}$ |
| 8.0 | 40 | 80 | 160 | $5.9\times10^{8}$ |
| 20.0 | 40 | 80 | 160 | $2.8\times10^{3}$ |

**Two things happen, and they are separate.**

**The count is exactly $m$ for every $pe > 0$, and exactly $m/2$ at $pe = 0$.** GMRES never
converges early on this family: the minimal polynomial has full degree, so exercise 2.3's bound
is attained. The halving at $pe = 0$ is exercise 26.2.4's condition: the symmetric operator with
a symmetric right-hand side excites only half the modes, so
$\dim K_m = n/2$ and GMRES finishes there.

**At $pe = 2$ the matrix is defective.** The superdiagonal is $-1 + pe/2$, which vanishes at
exactly $pe = 2$, leaving a lower bidiagonal matrix with constant diagonal 2: a **single Jordan
block** with no eigenvector basis at all. `eigenvector_conditioning` returns $\infty$, and
$\kappa(V)$ peaks there from both sides, $7\times10^{19}$ at $pe = 1.9$ and $1.2\times10^{41}$
at $pe = 2.1$.

**And $\kappa(V)$ comes back down.** At $pe = 20$ it is $2.8\times10^{3}$, only three orders of
magnitude, because the convection term now dominates so completely that the operator is nearly
lower triangular with well separated diagonal entries. **Non-normality is not monotone in the
Peclet number**, which is the finding most likely to surprise, and it is why lesson 27's
`departure_from_normality` docstring warns about reading the scale invariant measure as if it
were.

**The practical reading.** $pe = 2$ is the classical stability threshold for central
differences: above it the discrete solution oscillates. That the matrix is exactly defective
there is not a coincidence, it is the algebraic form of the same fact.

### 4.2 Orthogonality against the number of passes

This is measured in exercise 3.1's first table. Summarising by the smallest number of passes
that keeps the loss at roundoff:

| matrix | one pass (plain MGS) | two passes | three |
|---|---|---|---|
| well conditioned | $2.8\times10^{-13}$ at $m=55$ | $5.6\times10^{-16}$ | no further gain |
| nonnormal $\gamma=30$ | $2.8\times10^{-4}$ | $6.7\times10^{-16}$ | no further gain |
| ill conditioned $\kappa=10^{10}$ | $6.6\times10^{-8}$ | $6.7\times10^{-16}$ | no further gain |

**Two passes is enough, and it is enough for a reason rather than by luck.** Kahan and Parlett's
result, usually quoted as "twice is enough", says that if the first pass reduces the component
along the existing basis by any fixed factor, the second reduces it to roundoff. The rare cases
needing a third are ones where the vector was almost entirely in the existing space, and there
the right response is to declare a breakdown rather than reorthogonalize again, since the new
direction carries no information.

**That is exactly what `nalib`'s GMRES does**, one extra pass and a breakdown test, and this
measurement is why.

### 4.3 Which predictor

Measured over 21 matrices from the $\gamma$ family, $n = 40$, all with **identical**
eigenvalues $1,\dots,40$:

| predictor | correlation with the GMRES step count |
|---|---|
| $\log\kappa(V)$ | **0.801** |
| $\log\kappa(A)$ | 0.767 |
| distance from the origin to $W(A)$ | $-0.069$ |

with the step count ranging from 26 to 40.

**$\kappa(V)$ is the best of the three and it is not good enough.** A correlation of 0.80 over
a family this controlled means it explains about two thirds of the variance, and the residual
third is not noise: it includes the fact that mild non-normality **helps** while extreme
non-normality hurts, so the relationship is not even monotone.

**The distance to the field of values correlates at essentially zero**, which looks damning
until you notice why. On most of this family the origin is **inside** $W(A)$, so the distance
is exactly 0 and carries no information at all. It is not a bad predictor, it is an undefined
one over most of the range. That is the same limitation exercise 2.5 identified analytically.

**$\kappa(A)$ correlating at 0.77 is a red herring.** All these matrices have the same
eigenvalues, so $\kappa(A)$ varies only through the singular values, which move with $\gamma$
for the same reason $\kappa(V)$ does. It is measuring non-normality by proxy, not conditioning.

**So the honest answer to "is any of them good enough to use" is no.** For a nonsymmetric
matrix there is no cheap scalar that predicts GMRES convergence, and Theorem 27.4 says there
cannot be one built from the eigenvalues. What does describe it is the whole pseudospectrum,
which is expensive, and the practical substitute is to run a few iterations and look at the
residual curve.

### 5.1 Pseudospectra

**The definition.** For $\varepsilon > 0$,

$$\Lambda_\varepsilon(A) = \{z \in \mathbb{C} : \|(zI-A)^{-1}\|_2 > 1/\varepsilon\}
= \{z : \sigma_{\min}(zI-A) < \varepsilon\},$$

equivalently the set of $z$ that are eigenvalues of some $A + E$ with $\|E\| < \varepsilon$.
The three definitions agree, and the third is the one to think with.

For a **normal** matrix, $\sigma_{\min}(zI-A) = \min_i|z-\lambda_i|$, so
$\Lambda_\varepsilon$ is exactly the union of $\varepsilon$-discs around the eigenvalues: no
more information than the spectrum. For a nonnormal matrix it can be enormously larger.

**Computing it.** Evaluate $\sigma_{\min}(zI-A)$ on a grid and contour it. The cost is one SVD
per grid point, which is why it is a diagnostic rather than a routine.

```python
def pseudospectrum(A, real_range, imag_range, npts=80):
    """sigma_min(zI - A) on a grid. Sizes and ranges are all arguments."""
    n = A.shape[0]
    xs = np.linspace(*real_range, npts)
    ys = np.linspace(*imag_range, npts)
    out = np.empty((npts, npts))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            out[i, j] = np.linalg.svd((x + 1j * y) * np.eye(n) - A, compute_uv=False)[-1]
    return xs, ys, out
```

**What to expect on the $\gamma$ family.** At $\gamma = 0$ the pseudospectra are tight discs
around $1,\dots,40$. As $\gamma$ grows they merge into a single region that expands towards and
eventually **encloses the origin**, and the $\gamma$ at which that happens is close to the
$\gamma$ at which Elman's bound goes vacuous, which section 6 measured between $\gamma = 0$ and
$\gamma = 3$.

**The bound it supplies.** If $\Lambda_\varepsilon(A)$ has boundary $\Gamma$ of length $L$ at
distance $d$ from the origin, then

$$\frac{\|\mathbf{r}_k\|}{\|\mathbf{r}_0\|}
\le \frac{L}{2\pi\varepsilon}\max_{z \in \Gamma}\left|\frac{\varepsilon}{z}\right|^k
\cdot \text{(a factor from the contour)},$$

which is genuinely predictive where the eigenvalue bound is not, at the cost of an optimisation
over $\varepsilon$ and a contour integral.

**And the honest caveat.** Pseudospectral bounds are also not sharp, and there are matrices
where they too are loose by large factors. The claim is that they are the best available tool
that survives non-normality, not that they close the question.

### 5.2 The Greenbaum, Ptak and Strakos construction

**The theorem.** Given $f_0 \ge f_1 \ge \dots \ge f_{n-1} > f_n = 0$ and nonzero
$\lambda_1,\dots,\lambda_n \in \mathbb{C}$ closed under conjugation, there is a real matrix $A$
with those eigenvalues and a vector $\mathbf{b}$ such that GMRES applied to
$A\mathbf{x} = \mathbf{b}$ produces exactly $\|\mathbf{r}_k\| = f_k$.

**The construction.** Set $w_k = \sqrt{f_{k-1}^2 - f_k^2}$ for $k = 1,\dots,n$, which is
well defined by the monotonicity, and let $\mathbf{w} = (w_1,\dots,w_n)^T$, so
$\|\mathbf{w}\| = f_0$. Let $C$ be the companion matrix of the polynomial with roots
$\lambda_i$, and let $R$ be the upper triangular matrix with
$R_{ij} = \sqrt{f_{i-1}^2 - f_{n}^2}$ appropriately arranged so that $\mathbf{w} = R\mathbf{h}$
for the vector $\mathbf{h}$ of GMRES residual decrements. Then $A = WCW^{-1}$ and
$\mathbf{b} = W\mathbf{e}_1 f_0$ for a suitable $W$ built from $R$.

**Why it needs the curve strictly decreasing until the last step.** If $f_{k-1} = f_k$ for some
$k < n$, then $w_k = 0$, so $\mathbf{w}$ has a zero component and $R$ becomes singular. The
underlying reason is not a construction artefact: a step with zero residual decrement means
GMRES made **no** progress at that step, which happens only when the new Krylov direction
contributes nothing, and then the Krylov space has effectively stopped growing. The theorem is
usually stated with strict decrease for that reason, and the non-strict case is handled by a
limiting argument.

**A caution about implementing it.** The companion matrix of a polynomial with well separated
roots is famously ill conditioned: for roots $1,\dots,n$ the coefficients are of size $n!$, and
a naive implementation loses the eigenvalues completely. A first attempt at this returned
eigenvalues off by 0.91 at $n = 10$ while being exact at $n = 6$, which is the classic
signature. Building $A$ through a well conditioned similarity, or working with the roots
directly through a diagonal-plus-rank-one form, is necessary above about $n = 8$.

**Which is why lesson 27 uses the bidiagonal family instead.** It is a special case of the
theorem, its eigenvalues are the diagonal so they are exact by construction and checkable by
assertion, and it already settles the question the theorem is quoted for. A general
construction whose output you cannot verify proves nothing.

### 5.3 Transpose-free methods

**What each gives up.**

**CGS (conjugate gradient squared)** replaces BiCG's $A^T$ sequence by squaring the BiCG
polynomial: $\mathbf{r}_k = \phi_k(A)^2\mathbf{r}_0$ where BiCG has
$\phi_k(A)\mathbf{r}_0$. That removes $A^T$ and doubles the convergence rate when $\phi_k$ is
small, and **squares the oscillation** when it is not. What it gives up is any control at all
over the intermediate residual.

**BiCGSTAB** replaces the second $\phi_k$ by a product of one-dimensional minimisations,
$\mathbf{r}_k = \psi_k(A)\phi_k(A)\mathbf{r}_0$ with
$\psi_k(t) = \prod(1-\omega_i t)$ and each $\omega_i$ chosen to minimise the residual. That
smooths the curve, at the price of a second breakdown mode when $\omega_i \to 0$, which lesson
27 section 8 measured failing on all ten right-hand sides at $pe = 8$.

**TFQMR** applies a quasi-minimal residual smoothing to the CGS iterates, giving a
nearly monotone curve without any minimisation. It gives up the property that its iterates are
the ones a Galerkin condition would produce, so its residual is a smoothed proxy rather than
the real thing.

**Constructing a CGS failure.** The theory says
$\mathbf{r}_k^{\text{CGS}} = \phi_k(A)^2\mathbf{r}_0$ where BiCG has
$\phi_k(A)\mathbf{r}_0$, so the CGS peak should be about the **square** of the BiCG peak.
Measure both on the same problems and check.

```python
def cgs(A, b, tol=1e-10, max_iter=None):
    """Conjugate gradient squared: BiCG's polynomial applied twice, so no transpose is needed.

    The auxiliary vector q is not optional. Dropping it gives a recurrence that looks similar,
    runs without complaint, and converges nowhere.
    """
    b = np.asarray(b, dtype=float).ravel()
    n = b.size
    max_iter = 8 * n if max_iter is None else max_iter
    x = np.zeros(n)
    r = b.copy()
    r0 = r.copy()
    p = np.zeros(n)
    q = np.zeros(n)
    rho_prev = 1.0
    eps = np.finfo(float).eps
    residuals = [float(np.linalg.norm(r))]
    for k in range(max_iter):
        rho = float(r0 @ r)
        if abs(rho) <= eps * np.linalg.norm(r0) * np.linalg.norm(r):
            break
        if k == 0:
            u = r.copy()
            p = u.copy()
        else:
            beta = rho / rho_prev
            u = r + beta * q
            p = u + beta * (q + beta * p)
        v = A @ p
        d = float(r0 @ v)
        if abs(d) <= eps * np.linalg.norm(r0) * np.linalg.norm(v):
            break
        alpha = rho / d
        q = u - alpha * v
        w = u + q
        x = x + alpha * w
        r = r - alpha * (A @ w)
        residuals.append(float(np.linalg.norm(r)))
        rho_prev = rho
        if residuals[-1] <= tol * np.linalg.norm(b):
            break
    return x, np.array(residuals)
```

**Measured**, convection diffusion at $m = 100$, tolerance $10^{-9}$, peaks relative to the
starting residual:

| $pe$ | seed | BiCG peak | BiCG converged | CGS peak | CGS converged | ratio to BiCG$^2$ |
|---|---|---|---|---|---|---|
| 0.5 | 0 | 63.5 | yes | $9.2\times10^{3}$ | yes | 2.28 |
| 0.5 | 1 | 750.0 | yes | $1.7\times10^{6}$ | yes | 2.97 |
| 0.5 | 2 | 65.0 | yes | $8.2\times10^{3}$ | yes | 1.94 |
| **0.5** | **3** | **56114** | **yes** | **$6.8\times10^{9}$** | **NO** | 2.15 |
| 1.0 | 0 | 36.5 | yes | $3.3\times10^{3}$ | yes | 2.46 |
| 1.0 | 1 | 2759 | yes | $1.6\times10^{7}$ | yes | 2.06 |
| 2.5 | 1 | 120.5 | yes | $2.7\times10^{4}$ | yes | 1.85 |

**The squaring is real**: the median ratio of the CGS peak to the square of the BiCG peak is
**2.06** over the runs where both converge, so the theory is confirmed to a small constant.

**And the fourth row is the failure the exercise asks for.** BiCG's residual climbs to 56000
times its start and comes back down to $10^{-9}$ successfully. CGS's climbs to
$6.8\times10^{9}$, and at that point $6.8\times10^{9} \times u \approx 7\times10^{-7}$, so
**every digit below $10^{-7}$ has already been destroyed by cancellation**. It cannot reach
$10^{-9}$ afterwards no matter how the polynomial behaves, and it does not.

**That is why BiCGSTAB replaced CGS**, and the replacement was not about speed. Both cost two
matrix-vector products per step and neither is optimal. BiCGSTAB's local minimisation caps the
peak near BiCG's rather than at its square, measured at 415 against BiCG's 1875 on the section 8
problem, and the attainable accuracy follows directly.

**The general lesson is one lesson 05 already made**: a computation whose intermediate
quantities grow far beyond the final answer loses those digits permanently, whatever it does
afterwards. Here the intermediate quantity is a residual, the growth is measured in orders of
magnitude, and the lost digits are exactly the ones the tolerance was asking for.

---

## Lesson 28, Multigrid Methods

### 1.1 Why the residual is transferred and not the error

**Because the error is not available.** The error is $\mathbf{e} = \mathbf{x}^\ast -
\mathbf{x}$ and $\mathbf{x}^\ast$ is what you are computing. The residual
$\mathbf{r} = \mathbf{b} - A\mathbf{x}$ costs one matrix-vector product.

**And you lose nothing, because they solve the same equation.**

$$A\mathbf{e} = A\mathbf{x}^\ast - A\mathbf{x} = \mathbf{b} - A\mathbf{x} = \mathbf{r}.$$

So the residual equation $A\mathbf{e} = \mathbf{r}$ has the **same operator** as the original
problem, which means the coarse grid can be given the residual and asked the same kind of
question it would have been asked about the error.

**Two consequences that matter for the implementation.**

The coarse problem always starts from $\mathbf{x} = \mathbf{0}$, because the unknown there is a
**correction** rather than a solution, and there is no reason to prefer any nonzero starting
guess for it.

And the operator on the coarse grid must be a discretisation of the same differential operator,
not merely a smaller matrix. Lesson 28 section 4 checked this: $RAP$ equals the coarse
rediscretisation exactly for the model problem, and getting the $1/h^2$ scaling wrong makes the
coarse correction four times too large in 2D and stops the cycle converging altogether, which
is exactly what a first attempt at exercise 3.1 did.

### 1.2 Where the factor of 2 comes from

The grids are $n$, $n/2$, $n/4$, and so on, so one sweep on each level costs

$$n + \frac{n}{2} + \frac{n}{4} + \dots = n\sum_{k \ge 0}2^{-k} = 2n,$$

that is **2 fine grid sweeps** for one sweep on every level. With $\nu_1$ pre and $\nu_2$ post
sweeps the V-cycle costs $2(\nu_1+\nu_2)$ fine grid sweeps, which is where `work_units` comes
from.

**In 2D each level has a quarter of the points**, since both directions are halved:

$$1 + \frac14 + \frac1{16} + \dots = \frac{1}{1 - 1/4} = \frac43.$$

**In 3D it is an eighth**, giving $1/(1-1/8) = 8/7$.

**So the cost gets closer to a single fine grid sweep as the dimension rises**, which is the
opposite of the usual pattern and is one reason multigrid matters most where problems are
hardest. Measured in exercise 3.1: the ratio of total points to finest points is 1.258 at
$m = 15$, 1.295 at $m = 31$, 1.313 at $m = 63$, climbing towards $4/3 = 1.333$ as the number of
levels grows.

**A caution about the W-cycle.** It visits each coarse level twice, so the 1D series becomes
$1 + 2/2 + 4/4 + \dots$, which does **not** converge and gives $O(\log n)$ sweeps per cycle. In
2D it is $1 + 2/4 + 4/16 + \dots = 2$, which does. That is why W-cycles are affordable in 2D and
3D and awkward in 1D.

### 1.3 Twice as many cycles when $n$ doubles

**Two likely causes, and they are distinguishable by one measurement.**

**The smoother is not smoothing.** If the worst shrinkage over the upper half of the spectrum is
close to 1, the coarse grid is being handed an error it cannot represent and the cycle degrades
towards the smoother's own rate, which does grow with $n$. Lesson 28 section 7 measured plain
Jacobi failing at every size where damped Jacobi took 8 cycles.

**The coarse operator is not a discretisation of the same operator.** A scaling error, or a
rediscretisation that does not match $RAP$ for a variable coefficient problem, makes the
correction the wrong size. This is easy to do and gives exactly this symptom.

**How to tell them apart.** Compute `smoothing_factor(A)["worst_upper_half"]`. If it is above
about 0.6 the smoother is the problem. If it is 0.33 and the cycle still degrades, the transfer
or the coarse operator is.

**A third cause worth knowing.** Anisotropy: a point smoother only smooths in the strongly
coupled direction, so the error is smooth along one axis and rough along the other and neither
mechanism handles it. Exercise 4.3 measures this, and the diagnostic is that the convergence
factor degrades as the anisotropy grows rather than as $n$ grows.

### 2.1 Full weighting is $P^T/2$, and the coarse operator is symmetric

**The transfers, written out.** Linear interpolation puts coarse value $e_i$ at fine index
$2i+1$ and half of it at $2i$ and $2i+2$:

$$(Pe)_{2i+1} = e_i, \qquad (Pe)_{2i} = \tfrac12(e_{i-1} + e_i).$$

Full weighting takes

$$(Rr)_i = \tfrac14 r_{2i} + \tfrac12 r_{2i+1} + \tfrac14 r_{2i+2}.$$

**The comparison.** $P^T$ has $(P^T r)_i = \sum_j P_{ji}r_j = \tfrac12 r_{2i} + r_{2i+1} +
\tfrac12 r_{2i+2}$, which is exactly $2(Rr)_i$. So $R = P^T/2$, and lesson 28 section 4 checks
it numerically to $10^{-14}$ at every size.

**Symmetry of $RAP$.** With $R = cP^T$ for any scalar $c$,

$$(RAP)^T = (cP^TAP)^T = cP^TA^TP = cP^TAP = RAP$$

whenever $A$ is symmetric. **The scalar is irrelevant to symmetry**, but not to the method: it
sets the size of the coarse correction, and getting it wrong makes the cycle diverge while
leaving the matrix perfectly symmetric.

**And positive definiteness comes free.** For $\mathbf{v} \ne \mathbf{0}$,
$\mathbf{v}^TRAP\mathbf{v} = c(P\mathbf{v})^TA(P\mathbf{v}) > 0$ provided $P\mathbf{v} \ne
\mathbf{0}$, which holds because $P$ has full column rank (it maps $e_i$ to something nonzero
at fine index $2i+1$, and those indices are distinct). So the coarse problem is SPD whenever the
fine one is, at every level, which is what lets the recursion use the same machinery
throughout.

### 2.2 The coarse grid correction is an $A$-orthogonal projector

Write $T = I - P(RAP)^{-1}RA$, the operator taking an error to the error after the coarse grid
correction.

**It is a projector.** Let $S = P(RAP)^{-1}RA$. Then

$$S^2 = P(RAP)^{-1}\underbrace{RAP}_{}(RAP)^{-1}RA = P(RAP)^{-1}RA = S,$$

the middle $RAP$ cancelling its own inverse. So $S^2 = S$ and hence $T^2 = (I-S)^2 =
I - 2S + S^2 = I - S = T$.

**Its range and null space.** $S$'s range is contained in $\operatorname{range}(P)$, and for
$\mathbf{v} = P\mathbf{w}$ we get $S\mathbf{v} = P(RAP)^{-1}RAP\mathbf{w} = P\mathbf{w} =
\mathbf{v}$, so $\operatorname{range}(S) = \operatorname{range}(P)$ exactly. So $T$ annihilates
the coarse grid space and leaves its complement, and the complement is where the smoother has to
work. **That is the division of labour, stated as an algebraic fact.**

**$A$-orthogonality.** $S$ is self-adjoint in the $A$-inner product when
$\langle S\mathbf{u},\mathbf{v}\rangle_A = \langle \mathbf{u},S\mathbf{v}\rangle_A$ for all
$\mathbf{u},\mathbf{v}$, that is when $(S\mathbf{u})^TA\mathbf{v} = \mathbf{u}^TAS\mathbf{v}$,
that is when $S^TA = AS$. With $R = cP^T$ and $A$ symmetric,

$$AS = AP\left(cP^TAP\right)^{-1}cP^TA, \qquad
S^TA = cA^TP\left(cP^TA^TP\right)^{-1}P^TA = cAP\left(cP^TAP\right)^{-1}P^TA,$$

and the two agree. So $S$, and therefore $T$, is $A$-orthogonal **precisely when $R$ is a
scalar multiple of $P^T$**, which is what the variational condition of exercise 2.1 asks for.

**And that is what "variationally consistent" buys.** An $A$-orthogonal projector has
$\|T\|_A = 1$ and never increases the $A$-norm of the error, so the coarse grid correction can
never make things worse. With $R$ chosen independently of $P$ the projector is **oblique**, and
lesson 16 measured $\|P\|_2 = 1/\cos\theta_{\max}$ for those, which can be large. That is the
theoretical reason injection is inferior, and exercise 4.3 of lesson 28 measures the practical
size of it at about 25 percent more cycles.

### 2.3 The damped Jacobi smoothing factor

Jacobi on the model problem has $G_{\text{J}} = I - D^{-1}A$ with eigenvalues
$\cos(k\pi h)$, so damped Jacobi has

$$\lambda_k(\omega) = 1 - \omega\left(1 - \cos(k\pi h)\right).$$

The **smoothing factor** is the worst of these over the upper half of the spectrum, that is
$k > n/2$, equivalently $\theta = k\pi h \in [\pi/2, \pi)$:

$$\mu(\omega) = \max_{\pi/2 \le \theta < \pi}\left|1 - \omega(1-\cos\theta)\right|.$$

Substituting $s = 1 - \cos\theta$, which runs over $[1, 2)$, the quantity is
$\max_{1 \le s < 2}|1 - \omega s|$. The function $1 - \omega s$ is linear and decreasing in $s$
for $\omega > 0$, so its extremes over $[1,2]$ are at the endpoints:

$$\mu(\omega) = \max\left(|1-\omega|,\ |1-2\omega|\right).$$

**Minimising.** $|1-\omega|$ decreases on $(0,1]$ and $|1-2\omega|$ decreases then increases
with a minimum at $\omega = 1/2$. The maximum of the two is minimised where they cross with
opposite slopes: $1 - \omega = 2\omega - 1$, giving

$$\boxed{\omega = \tfrac23, \qquad \mu = \tfrac13.}$$

**Both facts confirmed.** Lesson 28 section 2 measures the worst upper-half shrinkage as exactly
0.3333 at $\omega = 2/3$, and section 7's sweep over $\omega$ finds nothing better: 0.7000 at
$\omega = 0.3$, 0.5000 at 0.5, **0.3333** at 2/3, 0.5998 at 0.8, 0.7997 at 0.9, and 0.9997 at 1.

**Note what the derivation did not use.** The size $n$ appears only through $h$, and $h$
cancelled when the endpoints were taken. **The smoothing factor is independent of the mesh**,
which is the property the whole method rests on: the smoother's own convergence rate degrades
with $n$ and its smoothing factor does not.

### 2.4 The two-grid factor is independent of $h$

**The setup.** For the model problem the sine modes come in pairs: mode $k$ and mode $n+1-k$
are indistinguishable on the coarse grid, since $\sin((n+1-k)\pi j h)$ sampled at even $j$
equals $\pm\sin(k\pi jh)$. So the two-grid operator does not couple different pairs, and it is
**block diagonal with $2\times2$ blocks**, one per pair $(k, n+1-k)$ for $k < (n+1)/2$.

**Each block.** Writing $c = \cos^2(k\pi h/2)$ and $s = \sin^2(k\pi h/2)$, so $c + s = 1$, the
coarse grid correction on the pair is

$$I - \begin{pmatrix} c \\ s \end{pmatrix}\begin{pmatrix} c & s\end{pmatrix}
= \begin{pmatrix} s & -s \\ -c & c \end{pmatrix}\cdot\text{(after simplification)},$$

and the smoother contributes $\operatorname{diag}(\lambda_k^\nu, \lambda_{n+1-k}^\nu)$ with
$\lambda_k = 1 - \tfrac23(1-\cos k\pi h)$. **The block depends on $k$ and $h$ only through the
combination $\theta = k\pi h$**, and as $k$ ranges over its values $\theta$ fills $(0,\pi)$
more and more densely without ever leaving it. So the maximum over $k$ of the block's spectral
radius converges to a supremum over $\theta \in (0,\pi)$ that does not depend on $h$ at all.

**That is the whole argument**, and it is worth separating from the algebra: the two-grid
factor is a maximum of a fixed function over a fixed interval, sampled at $n$ points. Refining
the mesh samples the same function more finely; it does not change the function.

**The closed form for $\nu_1 = \nu_2 = 2$.** Carrying the algebra through, the block spectral
radius as a function of $\theta$ is maximised where the smoothed and unsmoothed parts balance,
and the value is

$$\rho_{\text{TG}} = \frac{5}{81} = 0.0617283950617\ldots$$

**Measured, and this is the sharpest confirmation in Part 4:** lesson 28 section 5 reports
$\rho$ to twelve digits at $n = 7, 15, 31, 63, 127$ and it is $0.061728395062$ every time,
agreeing with $5/81$ to within $10^{-16}$. Meanwhile the smoother's own spectral radius climbs
$0.949, 0.987, 0.997, 0.9992, 0.9998$ over the same sizes.

### 2.5 The approximation and smoothing properties

**The smoothing property.** There is a function $\eta(\nu) \to 0$ with

$$\|AS^\nu\|_2 \le \eta(\nu)\,\|A\|_2$$

where $S$ is the smoother's error propagation operator. In words: after $\nu$ sweeps, whatever
error remains has a **small residual relative to the size of the operator**, that is it is
smooth.

**The approximation property.** There is a constant $C$ independent of $h$ with

$$\left\|A^{-1} - P A_c^{-1}R\right\|_2 \le \frac{C}{\|A\|_2}.$$

In words: the coarse grid inverse approximates the fine one to within the size of the operator,
so the coarse correction is a genuine approximate solve for the smooth part.

**Together they give a mesh-independent bound.** The two-grid operator is
$T = (I - PA_c^{-1}RA)S^\nu$, and

$$\|T\|_2 \le \left\|A^{-1} - PA_c^{-1}R\right\|_2\,\|AS^\nu\|_2
\le \frac{C}{\|A\|_2}\cdot\eta(\nu)\|A\|_2 = C\,\eta(\nu).$$

**Neither $h$ nor $n$ appears in the result**, and that is the entire content: the two
$h$-dependent factors are $\|A\|_2$ and $1/\|A\|_2$, and they cancel. Choosing $\nu$ large
enough that $C\eta(\nu) < 1$ gives convergence at a rate independent of the mesh.

**Which one fails for an anisotropic operator?** **The smoothing property.** For
$-\varepsilon u_{xx} - u_{yy}$ with small $\varepsilon$, a point smoother reduces the error
components that oscillate in $y$, the strongly coupled direction, and does essentially nothing
to those oscillating in $x$ alone, because the $x$-coupling is $\varepsilon$-weak and the
diagonal is dominated by the $y$-coupling. So $\|AS^\nu\|$ does not become small relative to
$\|A\|$: the leftover error is **not** smooth, it is smooth in one direction only.

The approximation property survives, because the coarse grid still approximates the operator
correctly. Exercise 4.3 confirms the diagnosis by fixing the smoother rather than the transfer:
a line smoother in the strong direction restores mesh independence completely.

### 3.1 A 2D V-cycle

Bilinear interpolation and its transpose, all matrix free, with $m$ a free parameter:

```python
def apply_2d(u, m, eps=1.0):
    """(-eps u_xx - u_yy) on an m by m grid, INCLUDING the 1/h^2 factor.

    The scaling is not cosmetic. A cycle compares residuals across levels, so every level must
    discretise the same differential operator. Leaving the stencil unscaled makes the coarse
    operator four times too small in 2D and the cycle stops converging entirely.
    """
    h2 = (m + 1.0) ** 2
    g = u.reshape(m, m)
    out = (2.0 * eps + 2.0) * g
    out[:, :-1] -= eps * g[:, 1:]
    out[:, 1:] -= eps * g[:, :-1]
    out[:-1, :] -= g[1:, :]
    out[1:, :] -= g[:-1, :]
    return out.ravel() * h2


def restrict_2d(r, m):
    """Full weighting in 2D: the (1,2,1) x (1,2,1) / 16 stencil. m -> (m-1)//2."""
    g = r.reshape(m, m)
    mc = (m - 1) // 2
    if mc < 1:
        return np.zeros(0), 0
    out = np.zeros((mc, mc))
    for i in range(mc):
        for j in range(mc):
            a, b = 2 * i + 1, 2 * j + 1
            out[i, j] = (4 * g[a, b]
                         + 2 * (g[a-1, b] + g[a+1, b] + g[a, b-1] + g[a, b+1])
                         + (g[a-1, b-1] + g[a-1, b+1] + g[a+1, b-1] + g[a+1, b+1])) / 16.0
    return out.ravel(), mc
```

**Measured**, damped Jacobi at $\omega = 2/3$, $\nu_1 = \nu_2 = 2$, tolerance $10^{-10}$:

| $m$ | $n = m^2$ | levels | cycles | factor | relative error |
|---|---|---|---|---|---|
| 7 | 49 | 2 | 12 | 0.1678 | $1.1\times10^{-10}$ |
| 15 | 225 | 3 | 13 | 0.1941 | $2.5\times10^{-10}$ |
| 31 | 961 | 4 | 13 | 0.1969 | $3.9\times10^{-10}$ |
| 63 | 3969 | 5 | 13 | 0.1995 | $4.4\times10^{-10}$ |

**Mesh independent, as in 1D**, at 13 cycles from $n = 49$ to $n = 3969$. The factor is 0.20
rather than 1D's 0.075, which is expected: 2D has more error components per unit of smoothing
and the coarse grid must represent a two-dimensional function.

**And the work per cycle:**

| $m$ | level sizes | total / finest |
|---|---|---|
| 15 | 225, 49, 9 | 1.258 |
| 31 | 961, 225, 49, 9 | 1.295 |
| 63 | 3969, 961, 225, 49, 9 | 1.313 |

climbing towards $4/3 = 1.333$ as the number of levels grows, exactly as exercise 1.2's
geometric series predicts, and well under 1D's factor of 2.

**The bug worth recording.** A first version omitted the $1/h^2$ factor in `apply_2d`,
reasoning that a constant scaling cannot matter for a linear solve. It does: the coarse
operator was then four times too small relative to the fine one, so the correction was four
times too large, and the measured convergence factors were 0.86, 0.95, 0.96, 0.96, that is no
convergence at all. **A cycle that compares quantities across grids has no freedom about
scaling**, unlike a single-level solver where any constant cancels.

### 3.2 Full multigrid

FMG starts on the coarsest grid, solves there, interpolates the **solution** upward as the
initial guess for the next level, and does one V-cycle. The gain is that each level starts from
an initial guess whose error is already at the coarse discretisation level.

```python
def fmg_1d(rhs_fn, n, nu1=2, nu2=2, cycles_per_level=1):
    """Full multigrid: start coarse and interpolate the SOLUTION upward, one V-cycle per level.

    rhs_fn(size) supplies the right-hand side on each grid, since the problem has to be
    discretised at every level rather than restricted. n is free and the levels follow from it.
    """
    sizes = [n]
    while sizes[-1] > 3 and (sizes[-1] - 1) // 2 >= 1:
        sizes.append((sizes[-1] - 1) // 2)
    x = None
    for size in reversed(sizes):
        A = mg.poisson_1d(size)
        x = np.zeros(size) if x is None else mg.prolong_linear(x, size)
        for _ in range(cycles_per_level):
            x = mg.v_cycle(A, rhs_fn(size), x, nu1, nu2, hierarchy=mg.coarse_operators(A))
    return x
```

**Measured** on $-u'' = \pi^2\sin(\pi t)$ with $u = \sin(\pi t)$, one V-cycle per level:

| $n$ | discretisation error | FMG total error | FMG algebraic error | V-cycles to match |
|---|---|---|---|---|
| 15 | $3.22\times10^{-3}$ | $2.18\times10^{-3}$ | $1.04\times10^{-3}$ | 2 |
| 31 | $8.04\times10^{-4}$ | $5.11\times10^{-4}$ | $2.95\times10^{-4}$ | 3 |
| 63 | $2.01\times10^{-4}$ | $1.25\times10^{-4}$ | $7.64\times10^{-5}$ | 3 |
| 127 | $5.02\times10^{-5}$ | $3.11\times10^{-5}$ | $1.93\times10^{-5}$ | 4 |
| 255 | $1.26\times10^{-5}$ | $7.75\times10^{-6}$ | $4.84\times10^{-6}$ | 4 |
| 511 | $3.14\times10^{-6}$ | $1.94\times10^{-6}$ | $1.21\times10^{-6}$ | 4 |

**FMG's algebraic error after one pass is below the discretisation error at every size.** That
is the claim, and it is the right criterion: solving the discrete system more accurately than
the discretisation error is wasted work, since the answer is wrong by the discretisation error
anyway.

**A plain V-cycle needs 4 cycles to reach the same place**, so FMG is about four times cheaper.
It is also slightly cheaper than that: FMG's single pass includes the coarse levels, which cost
almost nothing.

**Two things worth separating in that table.** The **total** error compares to the continuous
solution and is bounded below by the discretisation error. The **algebraic** error compares to
the discrete solution and is what a solver controls. FMG's total error is actually *smaller*
than the discretisation error here, $2.18\times10^{-3}$ against $3.22\times10^{-3}$, which
looks impossible until you notice that FMG's answer is not the discrete solution and its error
happens to partly cancel the discretisation error. That is a known FMG effect and it is a
coincidence of this problem, not something to rely on.

**The right claim is the algebraic one**: one FMG pass leaves an algebraic error safely below
the discretisation error, so there is nothing left worth computing.

### 3.3 A line smoother

For $-\varepsilon u_{xx} - u_{yy}$ with $\varepsilon$ small, the strong coupling is in $y$, so
solve whole $y$-lines exactly. Each line is tridiagonal, which is lesson 21's Thomas algorithm.

```python
def line_smooth_2d(b, x, m, sweeps, eps=1.0):
    """Solve each line in the STRONG direction exactly, with the Thomas algorithm.

    Cost per sweep is O(m^2), the same order as a point sweep, because a tridiagonal solve is
    linear in its size and there are m lines of m points.
    """
    h2 = (m + 1.0) ** 2
    for _ in range(sweeps):
        g = x.reshape(m, m).copy()
        rhs = b.reshape(m, m)
        for j in range(m):                       # column j is one line in y
            off = np.zeros(m)
            if j > 0:
                off += eps * g[:, j - 1] * h2
            if j < m - 1:
                off += eps * g[:, j + 1] * h2
            g[:, j] = thomas(np.full(m - 1, -h2), np.full(m, (2.0 * eps + 2.0) * h2),
                             np.full(m - 1, -h2), rhs[:, j] + off)
        x = g.ravel()
    return x
```

**Measured at $m = 31$**, cycles to a relative residual of $10^{-8}$, with the convergence
factor in brackets:

| $\varepsilon$ | point smoother | line smoother |
|---|---|---|
| 1 | 10 (0.189) | **5** (0.017) |
| $10^{-1}$ | 48 (0.756) | **4** (0.006) |
| $10^{-2}$ | never (0.960) | **3** (0.001) |
| $10^{-3}$ | never (0.984) | **2** (0.000) |
| $10^{-4}$ | never (0.987) | **1** (0.000) |

**The line smoother gets better as the anisotropy gets worse**, which is the opposite of the
point smoother and is not a coincidence. As $\varepsilon \to 0$ the operator decouples into
independent $y$-lines, and the line smoother solves each of them **exactly**, so at
$\varepsilon = 10^{-4}$ one cycle is enough.

**And it costs the same order.** A tridiagonal solve is $O(m)$ per line and there are $m$ lines,
so a line sweep is $O(m^2)$, the same as a point sweep. The constant is larger, roughly a factor
of 3, which is nothing against a factor of 10 or infinity in the cycle count.

**The general principle.** A smoother must be strong in whatever direction the operator is
strongly coupled. Point smoothing assumes the coupling is isotropic; when it is not, either
smooth along lines, or coarsen only in the weak direction (semicoarsening), or both. **The fix
is always to the smoother, never to the transfer**, because exercise 2.5 identified the
smoothing property as the one that fails.

### 4.1 The factor against the smoothing count

Measured at $n = 255$, $\nu_1 = \nu_2 = \nu/2$:

| $\nu = \nu_1+\nu_2$ | factor | $1/\nu$ | factor $\times\ \nu$ |
|---|---|---|---|
| 2 | 0.18434 | 0.5000 | 0.3687 |
| 4 | 0.07977 | 0.2500 | 0.3191 |
| 6 | 0.05114 | 0.1667 | 0.3069 |
| 8 | 0.03733 | 0.1250 | 0.2986 |
| 10 | 0.02862 | 0.1000 | 0.2862 |
| 12 | 0.02418 | 0.0833 | 0.2901 |
| 14 | 0.02097 | 0.0714 | 0.2936 |
| 16 | 0.01690 | 0.0625 | 0.2704 |

**Fitted exponent: $-1.127$, against the theoretical $-1$.** The last column, factor times
$\nu$, is nearly constant at 0.27 to 0.37, which is the same statement.

**Where the fit is imperfect and why.** The theoretical $O(1/\nu)$ comes from the smoothing
property $\|AS^\nu\| \le \eta(\nu)\|A\|$ with $\eta(\nu) \sim 1/\nu$ for damped Jacobi, and
that estimate is asymptotic in $\nu$. At $\nu = 2$ the measured factor is 0.184 while $1/\nu$
scaled by the fitted constant would give 0.147, so the small-$\nu$ end is where the fit is
loosest. At the other end the factor is dominated by the **approximation** property's constant
rather than the smoothing property, so more smoothing eventually stops helping: the drop from
$\nu = 14$ to $\nu = 16$ is only 19 percent for 14 percent more work.

**The practical reading, which is not the fit.** More smoothing always improves the factor and
always costs proportionally more, so total work is roughly flat and the choice is about
robustness. $\nu = 2$ gives 0.18 on this problem and would give something much worse on a
harder one; $\nu = 4$ gives 0.08 with margin to spare. That is why $\nu_1 = \nu_2 = 2$ is
conventional.

### 4.2 Work against the level count

The work per cycle in fine grid sweeps is
$\sum_{\ell}(n_\ell/n_0) \approx 2$ in 1D, and the measurement in exercise 3.1 tracks the 2D
version approaching $4/3$.

Measured in 1D, the total over all levels divided by the finest:

| levels | 1D ratio | 2D ratio |
|---|---|---|
| 2 | 1.500 | 1.250 |
| 3 | 1.750 | 1.312 |
| 4 | 1.875 | 1.328 |
| 5 | 1.938 | 1.332 |
| 8 | 1.992 | 1.333 |
| limit | **2** | **4/3** |

These are the idealised series. The measured 2D ratios in exercise 3.1 were 1.258, 1.295 and
1.313 at $m = 15, 31, 63$, slightly below these because the actual coarsening
$m \to (m-1)//2$ loses a point at every level rather than halving exactly.

**And the wall clock does not match, which is the interesting part.** Lesson 28 section 6
measures the matrix-free version at a flat 11 microseconds per unknown, so it does realise
$O(n)$. The **dense** implementation does not: building the coarse operators as $RAP$ costs
$O(n^3)$ once at setup, and each smoothing sweep is a dense $O(n^2)$ matrix-vector product
instead of an $O(n)$ stencil.

**The gap between the two is entirely implementation.** The algorithm's cost is counted in
sweeps and the sweeps are counted correctly by both. What differs is what a sweep costs, and a
dense sweep costs $n$ times what a stencil sweep costs. That is why lesson 28 keeps both: the
dense one so the algorithm is legible, the stencil one so the complexity claim is demonstrated
rather than asserted.

**A separate point about setup.** An early version rebuilt the coarse operators inside every
cycle, making the dense implementation $O(n^3)$ **per cycle** and taking lesson 28's notebook
144 seconds to run. Hoisting the construction into a setup phase, which is what every real
multigrid code does, brought it to 10 seconds. Setup and solve are separate for a reason.

### 4.3 Anisotropy

Measured, cycles to $10^{-8}$ with the convergence factor in brackets, point smoother:

| $\varepsilon$ | $m=15$ | $m=31$ | $m=63$ |
|---|---|---|---|
| 1 | 10 (0.188) | 10 (0.194) | 10 (0.191) |
| $10^{-1}$ | 47 (0.737) | 52 (0.767) | 52 (0.770) |
| $10^{-2}$ | never (0.923) | never (0.958) | never (0.966) |
| $10^{-3}$ | never (0.946) | never (0.983) | never (0.987) |
| $10^{-4}$ | never (0.949) | never (0.987) | never (0.989) |

**Mesh independence survives at $\varepsilon = 1$ and $\varepsilon = 0.1$** (10, 10, 10 and 47,
52, 52) and the factor is simply worse in the second case. **Below $\varepsilon = 10^{-2}$ the
method fails outright**, and the factor also starts growing with $m$: 0.923, 0.958, 0.966. So
two things go wrong together, and the anisotropy is the cause of both.

**The mechanism, mode by mode.** The operator's eigenvalues are
$\varepsilon(2-2\cos k_x\pi h) + (2 - 2\cos k_y\pi h)$. A mode oscillating in $x$ only has
eigenvalue $O(\varepsilon)$ while the diagonal is $2\varepsilon+2 \approx 2$, so damped Jacobi
shrinks it by $1 - \tfrac23\cdot O(\varepsilon) \approx 1$. **Those modes are invisible to the
smoother.** They are also poorly represented on the coarse grid, because they oscillate. Neither
mechanism touches them, and the complementarity that makes multigrid work is broken.

**The fix, measured in exercise 3.3**: a line smoother in the strong direction takes
$\varepsilon = 10^{-4}$ from "never" to **one cycle**. The alternative, semicoarsening, coarsens
only in the weak direction so the coarse grid can still represent what the smoother leaves; it
costs more levels and is preferred when the anisotropy direction varies across the domain, where
no single line direction is right.

### 5.1 The approximation property, measured

Estimate $C$ in $\|A^{-1} - PA_c^{-1}R\|_2 \le C/\|A\|_2$ directly:

```python
def approximation_constant(A, nc):
    """C in ||A^-1 - P Ac^-1 R|| <= C / ||A||. Sizes come from A and nc."""
    n = A.shape[0]
    P = np.column_stack([mg.prolong_linear(np.eye(nc)[:, i], n) for i in range(nc)])
    R = 0.5 * P.T
    Ac = R @ A @ P
    gap = np.linalg.inv(A) - P @ np.linalg.inv(Ac) @ R
    return float(np.linalg.norm(gap, 2) * np.linalg.norm(A, 2))
```

**Expected behaviour.** For the model problem $C$ should be a modest constant independent of
$n$, since that independence is the whole content of the property. For a variable coefficient
problem it grows with the coefficient jump, because the coarse grid represents a function that
is continuous across the jump while the true solution has a kink there.

**What to check, and how to check it honestly.** Compute $C$ at $n = 7, 15, 31, 63, 127$ and
confirm it is flat rather than merely bounded. Then repeat for the jumping coefficient family
of lesson 28 section 9 and see whether $C$ moves. If it does not, that is the algebraic reason
the cycle count did not move either, and it is a much better explanation than "the Galerkin
operator adapts".

**A caution about the measurement.** $\|A^{-1}\|$ grows like $h^{-2}$ and
$\|A\|$ like $h^{2}$... the other way round: $\|A\| \sim h^{-2}$ and $\|A^{-1}\| \sim 1$. The
product $C$ is therefore a difference of two large-ish quantities scaled by a large one, and it
should be computed with the SVD rather than by forming $\|A^{-1}\|$ separately. Forming
$\operatorname{inv}(A)$ at all is the sort of thing lesson 19 warns about, and it is acceptable
here only because this is a measuring instrument at small $n$, not a solver.

### 5.2 Algebraic multigrid

**The problem AMG solves.** Everything in lesson 28 assumed a grid that can be halved. A matrix
from an unstructured mesh has no such structure, and yet the same principle should apply: the
smoother leaves an error that is smooth **in the sense the matrix defines**, and a coarse space
should represent it.

**Classical Ruge-Stuben coarsening.** Define a **strong connection**: $i$ depends strongly on
$j$ when

$$-a_{ij} \ge \theta\max_{k \ne i}(-a_{ik}),$$

with $\theta$ typically 0.25. The reasoning is that algebraically smooth error varies slowly
along strong connections, since those are the directions the smoother has equilibrated.

Then choose coarse variables greedily: repeatedly pick the variable with the most strong
connections to undecided variables, make it coarse, make everything strongly connected to it
fine, and repeat. Two heuristics govern the choice: every fine variable should be strongly
connected to at least one coarse variable, and coarse variables should not be strongly
connected to each other.

**Interpolation.** For a fine variable $i$, distribute its value from its strongly connected
coarse neighbours, weighting by the connection strength and folding the strongly connected
**fine** neighbours into those weights. Then $R = P^T$ and $A_c = P^TAP$, the Galerkin form,
which is now mandatory: there is no grid to rediscretise on.

```python
def strong_connections(A, theta=0.25):
    """The strong connection graph. Sizes and the pattern both come from A."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    S = np.zeros((n, n), dtype=bool)
    for i in range(n):
        off = -A[i].copy()
        off[i] = -np.inf
        biggest = off.max()
        if biggest > 0:
            S[i] = off >= theta * biggest
        S[i, i] = False
    return S
```

**What to compare, and what to expect.** On the 1D model problem AMG should rediscover the
standard coarsening, choosing every other point, and produce a convergence factor close to
geometric multigrid's. That is the check that the implementation is right, because the answer is
known.

Then run it on something unstructured. **Expect it to be worse than geometric multigrid and
much better than anything else**, with a factor around 0.2 to 0.5 rather than 0.075, and a
setup cost that is a real fraction of the solve, sometimes larger than a single solve.

**And expect it to be less reliable.** AMG's heuristics are calibrated for M-matrices from
elliptic operators, which is lesson 25 exercise 2.5's class again. Outside it, the strong
connection definition can pick the wrong variables and the factor degrades without warning.

### 5.3 Optimality, and what it does not mean

**What $O(n)$ promises.** The number of arithmetic operations grows linearly with the number of
unknowns. Since reading the right-hand side is already $O(n)$, no method can do better in order,
so multigrid is optimal in the only sense complexity theory offers.

**Four things it does not promise.**

**The constant.** Measured, the matrix-free 1D V-cycle against the Thomas algorithm from
lesson 21, both $O(n)$:

| $n$ | Thomas | per unknown | V-cycle | per unknown | ratio |
|---|---|---|---|---|---|
| 4095 | 4.6 ms | 1.13 us | 54 ms | 13.2 us | 11.7 |
| 16383 | 16.9 ms | 1.03 us | 207 ms | 12.6 us | 12.2 |
| 65535 | 67.8 ms | 1.03 us | 802 ms | 12.2 us | 11.8 |
| 262143 | 269 ms | 1.03 us | 3.25 s | 12.4 us | 12.1 |

**A flat factor of 12, and Thomas is the one written as a Python loop.** Both are linear in
$n$, and in 1D the direct method wins at every size by a constant that never goes away. Being
optimal in order settles nothing about which method to use.

**Memory traffic.** A V-cycle sweeps the finest grid four times per cycle and touches every
level, so it is bandwidth bound in the sense of lesson 08's roofline. The arithmetic intensity
of a stencil sweep is about 0.5 flops per byte, far to the left of the ridge, so the achieved
fraction of peak is small and no amount of algorithmic optimality changes it.

**Parallel scalability.** The coarse grids have few points and cannot occupy many processors, so
the coarsest levels are a serial bottleneck. At scale, the time is dominated by communication on
levels that hold almost no data. This is why parallel multigrid codes stop coarsening early and
solve a moderately sized coarse problem directly, and why the practical scaling is not the
theoretical one.

**Setup cost, for AMG.** Building the hierarchy requires choosing coarse variables and forming
$P^TAP$ at every level, which can cost as much as several solves. For a single solve AMG can
lose to a direct method that is asymptotically far worse.

**The crossover with a direct solver.** For the 1D model problem, banded Cholesky is $O(n)$ with
a tiny constant and beats multigrid at every size. For the 2D model problem, nested dissection
gives $O(n^{3/2})$ flops and $O(n\log n)$ fill, so multigrid wins above roughly
$n \approx 10^4$ to $10^5$, depending on the constants. For 3D, direct is $O(n^2)$ with
$O(n^{4/3})$ fill and multigrid wins from a few thousand unknowns.

**What actually decides it is memory, not flops.** The direct method's fill has to fit, and when
it does not there is no crossover to compute: the direct method simply cannot run. That is the
same conclusion lesson 24 exercise 5.3 reached about CG against Cholesky, and it is the honest
summary of the whole of Part 4.

---

## Where these solutions sit in the course

Part 4 changed what a solver is allowed to know about a matrix.

- **Only the matrix-vector product**, which is why $n = 10^6$ fits in 32 MB and a dense
  factorization of the same problem would need 8 TB.
- **Convergence as a spectral question**: $\rho(G)$ for the stationary methods, the shape of
  the spectrum for Krylov ones. Not $n$, not sparsity, not the entries.
- **Clustering beats $\kappa$**, measured at 200 iterations against 14 for identical condition
  numbers, and that single fact is what preconditioning aims at and what deflation exploits.
- **Roundoff as a first order effect, not a correction.** Loss of orthogonality is not a small
  error in Lanczos and CG, it changes what the methods do, and Greenbaum's analysis explains
  the change rather than bounding it away.
- **The gap Part 4 leaves open**: every preconditioner measured here reduces the constant or at
  best halves the exponent, and none makes the iteration count independent of $n$. Lesson 28
  closes it, and it is the only method in the course that does.
