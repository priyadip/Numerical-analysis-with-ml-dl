# Solutions: Part 3, Direct Methods for Linear Systems

Worked solutions for the exercises in lessons 15 to 22.

Levels 1 and 2 are answered in full. Levels 3 and 4 give the method, the key code, and the
result you should get, so you can check your own work rather than copy it. Level 5 questions
are open ended, so those get a route through the problem and the answer where there is a
definite one.

Every number quoted was measured by running the code, not estimated.

---

## Lesson 15, Vectors, Matrices and Norms

### 1.1 Why $\operatorname{rank}(AB) \le \min(\operatorname{rank} A, \operatorname{rank} B)$

Column $j$ of $AB$ is $A\mathbf{b}_j$, which lies in the range of $A$. So every column of $AB$
lies in $\operatorname{range}(A)$, hence $\operatorname{range}(AB) \subseteq
\operatorname{range}(A)$ and $\operatorname{rank}(AB) \le \operatorname{rank}(A)$.

For the other half, apply the same argument to the transpose: $\operatorname{rank}(AB) =
\operatorname{rank}((AB)^T) = \operatorname{rank}(B^TA^T) \le \operatorname{rank}(B^T) =
\operatorname{rank}(B)$.

### 1.2 $\kappa_2 = 20$, and the worst amplification

$\kappa_2(A) = \|A\|_2\|A^{-1}\|_2 = 5 \times 4 = 20$.

A relative input error of $10^{-8}$ can become at most $20 \times 10^{-8} = 2\times10^{-7}$.
That is a loss of about $\log_{10}20 = 1.3$ digits, which is nothing. A condition number of 20
is excellent.

### 1.3 Why $\|I\|_F = \sqrt{n}$ settles it

Every **induced** norm satisfies

$$\|I\| = \max_{\|x\|=1}\|Ix\| = \max_{\|x\|=1}\|x\| = 1.$$

The Frobenius norm of $I_n$ is $\sqrt{n}$, which exceeds 1 for $n \ge 2$. So no vector norm
induces it. One counterexample is enough to settle an "if and only if" in the negative
direction.

### 2.1 $\|A\|_\infty$ is the maximum absolute row sum

**Upper bound.** For any $\mathbf{x}$ with $\|\mathbf{x}\|_\infty = 1$,

$$|(A\mathbf{x})_i| = \left|\sum_j a_{ij}x_j\right| \le \sum_j |a_{ij}||x_j|
\le \sum_j |a_{ij}| \le \max_k \sum_j |a_{kj}|,$$

so $\|A\mathbf{x}\|_\infty \le \max_k\sum_j|a_{kj}|$ and hence $\|A\|_\infty$ is at most the
maximum row sum.

**Attained.** Let $k$ be a row achieving the maximum, and choose

$$x_j = \operatorname{sign}(a_{kj}), \quad\text{with } x_j = 1 \text{ when } a_{kj} = 0.$$

Then $\|\mathbf{x}\|_\infty = 1$ and $(A\mathbf{x})_k = \sum_j |a_{kj}|$, the maximum row sum.
So the bound is attained and the two are equal. $\square$

Both halves are needed. The first says the norm is no bigger; the second says it is no smaller.

### 2.2 $\|A\|_1 = \|A^T\|_\infty$

$\|A\|_1$ is the maximum absolute **column** sum of $A$, and $\|A^T\|_\infty$ is the maximum
absolute **row** sum of $A^T$. Row $i$ of $A^T$ is column $i$ of $A$, so the two maximisations
are over the same collection of numbers. $\square$

### 2.3 $\rho(A) \le \|A\|$ for every induced norm

Let $\lambda$ be an eigenvalue with $|\lambda| = \rho(A)$ and $\mathbf{v} \ne \mathbf{0}$ a
corresponding eigenvector. Then

$$\rho(A)\|\mathbf{v}\| = |\lambda|\,\|\mathbf{v}\| = \|\lambda\mathbf{v}\|
= \|A\mathbf{v}\| \le \|A\|\,\|\mathbf{v}\|.$$

Divide by $\|\mathbf{v}\| > 0$. $\square$

For complex $\lambda$ this is done in $\mathbb{C}^n$ with the induced complex norm; the real
case follows by taking real and imaginary parts.

### 2.4 $\|A\|_2 \le \sqrt{\|A\|_1\|A\|_\infty}$

$\|A\|_2^2 = \rho(A^TA)$, and by exercise 2.3 with the $\infty$-norm,

$$\rho(A^TA) \le \|A^TA\|_\infty \le \|A^T\|_\infty\|A\|_\infty = \|A\|_1\|A\|_\infty,$$

using submultiplicativity and exercise 2.2. Take square roots. $\square$

**How tight is it?** Measured over 240 random matrices at $n = 3, 10, 50, 200$, the ratio
$\|A\|_2 / \sqrt{\|A\|_1\|A\|_\infty}$ ranged from **0.146 to 0.882**. Never above 1, as the
theorem requires, and typically a factor of 2 to 5 loose.

That is still useful: it gives an $O(n^2)$ upper bound on the $O(n^3)$ quantity, which is
exactly the kind of cheap surrogate lesson 22 builds on.

### 2.5 The 2-norm is transpose invariant, the others are not

$\|A\|_2^2 = \rho(A^TA)$ and $\|A^T\|_2^2 = \rho(AA^T)$. The nonzero eigenvalues of $A^TA$ and
$AA^T$ coincide, because if $A^TA\mathbf{v} = \lambda\mathbf{v}$ with $\lambda \ne 0$ then
$AA^T(A\mathbf{v}) = \lambda(A\mathbf{v})$ and $A\mathbf{v} \ne 0$. So the spectral radii agree
and $\|A\|_2 = \|A^T\|_2$.

The deeper reason is that the 2-norm is defined through the **singular values**, and the SVD of
$A^T$ is the SVD of $A$ with $U$ and $V$ exchanged, leaving $\Sigma$ untouched (lesson 41).

$\|A\|_1 \ne \|A\|_\infty$ in general because they measure different things: columns against
rows. Any matrix with unequal row and column sums works, for instance
$\left(\begin{smallmatrix}1&1\\0&0\end{smallmatrix}\right)$, with $\|A\|_1 = 1$ and
$\|A\|_\infty = 2$.

### 3.1 $\|A\|_2$ by the power method

$\|A\|_2 = \sqrt{\rho(A^TA)}$, and the power method finds the dominant eigenvalue:

```python
def two_norm_power(A, iters=200, tol=1e-12, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    v = rng.standard_normal(A.shape[1])
    v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = A.T @ (A @ v)                # never form A^T A
        new = np.linalg.norm(w)
        if new == 0:
            return 0.0
        v = w / new
        if abs(new - lam) <= tol * new:
            break
        lam = new
    return np.sqrt(lam)
```

**Never form $A^TA$.** It costs $O(n^3)$ and squares the condition number (lesson 32). The
product $A^T(A\mathbf{v})$ is two matrix-vector products at $O(n^2)$.

At $n = 1000$ this is far faster than a full SVD, because it computes one singular value rather
than all of them. It converges slowly when $\sigma_1$ and $\sigma_2$ are close, at a rate
$(\sigma_2/\sigma_1)^2$ per step. Lesson 36 develops the power method properly.

### 3.2 The vector attaining the norm

For $p = \infty$, take the maximising row $k$ and set $x_j = \operatorname{sign}(a_{kj})$, as in
exercise 2.1. For $p = 1$, take the maximising **column** $k$ and set $\mathbf{x} =
\mathbf{e}_k$: then $A\mathbf{e}_k$ is column $k$, whose 1-norm is the maximum column sum.

```python
def norm_and_witness(A, p):
    if p == 1:
        k = int(np.argmax(np.abs(A).sum(axis=0)))
        x = np.zeros(A.shape[1]); x[k] = 1.0
    else:
        k = int(np.argmax(np.abs(A).sum(axis=1)))
        x = np.sign(A[k]); x[x == 0] = 1.0
    return np.linalg.norm(A @ x, p) / np.linalg.norm(x, p), x
```

The ratio equals the norm to machine precision in both cases, because both constructions are
exact rather than approximate.

### 3.3 The overflow in the naive 2-norm

`nalib.linalg.vector_norm(x, 2)` computes $\sqrt{\sum x_i^2}$ directly. With $x_i \approx
10^{200}$ the squares are $10^{400}$, which overflows to `inf`, and the answer is `inf` even
though $\|x\|_2 \approx 10^{200}$ is perfectly representable.

The fix is to scale by the largest entry first:

```python
def scaled_two_norm(x):
    m = np.max(np.abs(x))
    if m == 0.0:
        return 0.0
    return float(m * np.sqrt(np.sum((x / m) ** 2)))
```

Now every term is at most 1, so nothing overflows, and the same trick prevents underflow for
tiny entries. This is what LAPACK's `nrm2` does and why `numpy.linalg.norm` succeeds where the
naive version fails.

The cost is one extra pass over the data, which is why the naive version exists at all: it is
faster when you know the range of your data.

### 4.1 $\|A\|_F / \|A\|_2$ against $\sqrt{n}$

The bound is $\|A\|_F \le \sqrt{r}\|A\|_2$ with $r$ the rank, so for a full rank $n \times n$
matrix the bound is $\sqrt{n}$.

For **random Gaussian** matrices the typical ratio grows like $\sqrt{n}$ but with a constant
well below 1, because the singular values of a random matrix follow the Marchenko-Pastur
distribution and are spread out rather than equal. The ratio $\|A\|_F/\|A\|_2$ tends to about
$\sqrt{n}/2$ for large $n$.

The bound is attained exactly when all singular values are equal, meaning $A$ is a multiple of
an orthogonal matrix. Lesson 15 section 5 measured $\|I_6\|_F/\|I_6\|_2 = \sqrt{6}$ exactly.

### 4.2 Sharpness of submultiplicativity against dimension

Measure $\|AB\|/(\|A\|\|B\|)$ for random $A, B$. The ratio **decreases** with $n$, so the bound
gets **looser** in higher dimensions.

The reason is that $\|A\|$ is the stretching in the single worst direction. For the product to
attain $\|A\|\|B\|$, the direction $B$ stretches most must be the direction $A$ stretches most.
In high dimensions two independently chosen directions are nearly orthogonal, so the alignment
essentially never happens.

This is the same curse of dimensionality that made random sampling fail in lesson 15 section 4.

### 4.3 The three matmul views timed

The ordering is reliably

$$\text{outer products} \approx \text{by columns} \ \ll\ \text{inner products}$$

for a Python-level implementation, and all three are far slower than `A @ B`.

The inner product view is worst because it accesses `B[:, j]`, a **column** of a row-major
array, so consecutive elements are `n` floats apart in memory and every access is a cache miss.
The column and outer product views work with contiguous data.

This is lesson 08's finding again: the flop count is identical for all three, and the runtime is
not, because memory access dominates.

### 5.1 The numerical radius

$r(A) = \max_{\|x\|=1}|x^*Ax|$.

- $\rho(A) \le r(A)$: take $\mathbf{x}$ the unit eigenvector for the dominant eigenvalue, then
  $|x^*Ax| = |\lambda|$.
- $r(A) \le \|A\|_2$: by Cauchy-Schwarz, $|x^*Ax| \le \|x\|\|Ax\| \le \|A\|_2$.
- $\|A\|_2 \le 2r(A)$: from the polarisation identity, writing $y^*Ax$ in terms of quadratic
  forms.

**Attaining each.** The first is equality for **normal** matrices. The third is attained by
$\left(\begin{smallmatrix}0&1\\0&0\end{smallmatrix}\right)$, which has $\rho = 0$, $r = 1/2$
and $\|A\|_2 = 1$.

**What $r$ measures that neither other does.** $r(A) < 1$ implies $\|A^k\|$ is bounded for all
$k$, which $\rho(A) < 1$ does not give without a constant, and which $\|A\|_2 < 1$ gives but is
far too strong. The numerical radius controls the **transient** behaviour of section 6, which
is exactly what the spectral radius misses and the norm overstates.

### 5.2 Absolute and monotone norms are the same thing

**Monotone implies absolute.** If $|x_i| = |y_i|$ for all $i$ then each bounds the other, so
$\|x\| \le \|y\|$ and $\|y\| \le \|x\|$, hence equality. So the norm depends only on the
absolute values.

**Absolute implies monotone.** Suppose $|x_i| \le |y_i|$ for all $i$. Write $x = Dy$ where $D$
is diagonal with $|d_i| \le 1$. Then $D$ is a convex combination of diagonal sign matrices, and
an absolute norm is invariant under those, so by the triangle inequality $\|x\| = \|Dy\| \le
\|y\|$. $\square$

Every $p$-norm in this lesson is absolute and monotone. The "$p = 0.5$ norm" is absolute too,
which is why the axiom check caught it on the triangle inequality rather than there.

### 5.3 How far $\|A\|_2$ and $\rho(A)$ can separate

For a **normal** matrix ($A^*A = AA^*$) they are equal, because a normal matrix is unitarily
diagonalisable and unitary transformations preserve the 2-norm.

For non-normal matrices the ratio is unbounded. The family

$$A_t = \begin{pmatrix} 0 & t \\ 0 & 0\end{pmatrix}$$

has $\rho = 0$ and $\|A\|_2 = t$, so the ratio is infinite for every $t > 0$.

A more instructive family is the one from section 6, $\left(\begin{smallmatrix}0.9 & t\\ 0 &
0.8\end{smallmatrix}\right)$: $\rho$ stays at 0.9 while $\|A\|_2$ grows with $t$, and the
transient hump grows with it. The connection is direct: the size of the hump is controlled by
the **departure from normality**, and pseudospectra (lesson 40) are the tool that quantifies
it.

---

## Lesson 16, Orthogonality and Projectors

### 1.1 Why $\kappa \ge 1$ always

$\kappa(A) = \|A\|\|A^{-1}\| \ge \|AA^{-1}\| = \|I\| = 1$, using submultiplicativity and the
fact that every induced norm gives $\|I\| = 1$. $\square$

So $\kappa_2(Q) = 1$ for orthogonal $Q$ is not merely small, it is **the smallest value the
quantity can take**.

### 1.2 Rank 7, complement rank 13

A projector's eigenvalues are 0 and 1 only (exercise 2.1), so its trace equals the number of
eigenvalues equal to 1, which is its rank. Trace 7 means rank 7.

$I - P$ is also a projector, and $\operatorname{trace}(I - P) = 20 - 7 = 13$, so its rank is
13. The two ranks sum to the dimension, which says every vector splits uniquely between the two
subspaces.

### 1.3 An orthogonal projector is never an orthogonal matrix (unless $P = I$)

An orthogonal matrix is invertible, with $\kappa_2 = 1$. A projector onto a proper subspace is
**singular**: it sends the complement to zero. The only projector that is invertible is $P = I$,
which projects onto everything.

The word "orthogonal" is doing two different jobs. For a matrix it means $Q^TQ = I$; for a
projector it means the projection is *along a direction perpendicular to* the target subspace,
which is the condition $P^T = P$.

### 2.1 A projector's eigenvalues are 0 and 1, and its trace is its rank

**Eigenvalues.** If $P\mathbf{v} = \lambda\mathbf{v}$ with $\mathbf{v} \ne 0$, apply $P$ again:

$$\lambda\mathbf{v} = P\mathbf{v} = P^2\mathbf{v} = P(\lambda\mathbf{v}) = \lambda^2\mathbf{v}.$$

So $\lambda^2 = \lambda$, giving $\lambda \in \{0, 1\}$. $\square$

**Trace.** $P$ is diagonalisable, because the minimal polynomial $x^2 - x = x(x-1)$ has distinct
roots. So $P = S\Lambda S^{-1}$ with $\Lambda$ diagonal holding only 0s and 1s. The trace is
similarity invariant, so

$$\operatorname{trace}(P) = \operatorname{trace}(\Lambda) = \#\{\lambda_i = 1\}
= \operatorname{rank}(\Lambda) = \operatorname{rank}(P). \qquad\square$$

### 2.2 $P$ is an orthogonal projector if and only if $\|P\|_2 \le 1$

**Forward** is Theorem 16.8.

**Converse.** Let $P^2 = P$ with $\|P\|_2 \le 1$. Take $\mathbf{x} \in \operatorname{range}(P)$
and $\mathbf{y} \in \ker(P)$, and consider $\mathbf{x} + t\mathbf{y}$ for real $t$. Since
$P\mathbf{x} = \mathbf{x}$ and $P\mathbf{y} = 0$,

$$\|\mathbf{x}\|^2 = \|P(\mathbf{x} + t\mathbf{y})\|^2 \le \|\mathbf{x} + t\mathbf{y}\|^2
= \|\mathbf{x}\|^2 + 2t\,\mathbf{x}^T\mathbf{y} + t^2\|\mathbf{y}\|^2.$$

So $0 \le 2t\,\mathbf{x}^T\mathbf{y} + t^2\|\mathbf{y}\|^2$ for **every** real $t$, including
negative $t$ of small magnitude. Dividing by $t$ and letting $t \to 0$ from each side forces
$\mathbf{x}^T\mathbf{y} = 0$.

So the range and kernel are orthogonal, which is exactly what $P^T = P$ says. $\square$

This is the precise sense in which "norm 1" and "orthogonal" are the same condition, and it is
why section 7's measurement of $\|P\|_2$ is a genuine test rather than a symptom.

### 2.3 Orthogonal matrices form a group

**Closure.** $(Q_1Q_2)^T(Q_1Q_2) = Q_2^TQ_1^TQ_1Q_2 = Q_2^TIQ_2 = I$.

**Identity.** $I^TI = I$.

**Inverses.** $Q^{-1} = Q^T$, and $(Q^T)^TQ^T = QQ^T = I$ for square $Q$, so the inverse is
orthogonal too.

**Associativity** is inherited from matrix multiplication. $\square$

This is $O(n)$, the orthogonal group. The subgroup with $\det = +1$ is $SO(n)$, the rotations.

### 2.4 Deriving the Householder reflector

We want $H\mathbf{x} = \alpha\mathbf{e}_1$ with $H = I - 2\mathbf{v}\mathbf{v}^T/(\mathbf{v}^T\mathbf{v})$.

Since $H$ is orthogonal it preserves length, so $|\alpha| = \|\mathbf{x}\|$, giving
$\alpha = \pm\|\mathbf{x}\|$.

A reflector maps $\mathbf{x}$ to $\alpha\mathbf{e}_1$ when $\mathbf{v}$ is parallel to the
difference:

$$\mathbf{v} = \mathbf{x} - \alpha\mathbf{e}_1 = \mathbf{x} \mp \|\mathbf{x}\|\mathbf{e}_1.$$

Verify directly: with $\mathbf{v} = \mathbf{x} - \alpha\mathbf{e}_1$,

$$\mathbf{v}^T\mathbf{v} = \|\mathbf{x}\|^2 - 2\alpha x_1 + \alpha^2 = 2(\|\mathbf{x}\|^2 - \alpha x_1),
\qquad \mathbf{v}^T\mathbf{x} = \|\mathbf{x}\|^2 - \alpha x_1,$$

so $\mathbf{v}^T\mathbf{v} = 2\,\mathbf{v}^T\mathbf{x}$ and

$$H\mathbf{x} = \mathbf{x} - \frac{2\mathbf{v}(\mathbf{v}^T\mathbf{x})}{\mathbf{v}^T\mathbf{v}}
= \mathbf{x} - \mathbf{v} = \alpha\mathbf{e}_1. \qquad\square$$

**Both signs work mathematically**, and only one works numerically. Section 3 measured the
cancelling choice losing up to $5\times10^{-10}$ relative accuracy and dividing by exactly zero
when $\mathbf{x}$ is already on the axis.

### 2.5 $\|P\|_2 = 1/\cos\theta_{\max}$

Let $Q_A, Q_B$ be orthonormal bases for the two subspaces, so $P = Q_A(Q_B^TQ_A)^{-1}Q_B^T$.

Since $Q_A$ and $Q_B$ have orthonormal columns, they preserve norms, so

$$\|P\|_2 = \|(Q_B^TQ_A)^{-1}\|_2 = \frac{1}{\sigma_{\min}(Q_B^TQ_A)}.$$

The singular values of $Q_B^TQ_A$ are the **cosines of the principal angles** between the two
subspaces, by definition. The smallest of them is $\cos\theta_{\max}$, giving

$$\|P\|_2 = \frac{1}{\cos\theta_{\max}}. \qquad\square$$

Section 7 verified this to a relative $10^{-8}$ across six orders of magnitude.

### 3.1 Gram-Schmidt and what you find

```python
def gram_schmidt(A):
    A = np.asarray(A, dtype=float)
    Q = np.zeros_like(A)
    for j in range(A.shape[1]):
        v = A[:, j].copy()
        for i in range(j):                       # subtract (I - q q^T) for each earlier q
            v -= (Q[:, i] @ A[:, j]) * Q[:, i]
        Q[:, j] = v / np.linalg.norm(v)
    return Q
```

Measuring $\|Q^TQ - I\|$ against $\kappa(A)$ gives a **slope near 2** on a log-log plot: the
loss of orthogonality grows like $\kappa^2$, not $\kappa$.

That is the classical Gram-Schmidt failure, and it is one of the sharpest results in the
subject. The fix is a one-line change (subtract from the **running** $v$ rather than from the
original column), which is modified Gram-Schmidt and has slope 1. Lesson 30 measures both.

### 3.2 Householder QR without forming $H$

```python
def householder_qr(A):
    A = np.array(A, dtype=float)
    m, n = A.shape
    R = A.copy()
    Q = np.eye(m)
    for k in range(min(m - 1, n)):
        x = R[k:, k]
        v = x.copy()
        v[0] += np.sign(x[0]) * np.linalg.norm(x) if x[0] != 0 else np.linalg.norm(x)
        vv = v @ v
        if vv == 0:
            continue
        R[k:, k:] -= 2.0 * np.outer(v, v @ R[k:, k:]) / vv     # apply, never form H
        Q[:, k:] -= 2.0 * np.outer(Q[:, k:] @ v, v) / vv
    return Q, np.triu(R)
```

The key line applies the reflector as $\mathbf{x} - 2\mathbf{v}(\mathbf{v}^T\mathbf{x})/(\mathbf{v}^T\mathbf{v})$,
costing $O(mn)$ per column rather than the $O(m^2n)$ that forming $H$ would cost.

Both $\|A - QR\|/\|A\|$ and $\|Q^TQ - I\|$ come out at $O(u)$ regardless of $\kappa(A)$, which
is the property classical Gram-Schmidt lacks.

### 3.3 A tolerance that scales with $n$

$\|Q^TQ - I\|$ for `numpy.linalg.qr` grows roughly like $\sqrt{n}\,u$, because the errors in
the $n$ inner products accumulate in a random-walk fashion rather than adding coherently.

So the right tolerance is `tol = c * sqrt(n) * u` with $c$ around 10 to 100. A fixed `1e-12`
either rejects large valid matrices or accepts small invalid ones.

### 4.1 Determinants of random orthogonal matrices

With the sign correction, $\det Q = \pm 1$ with each sign about half the time, and the
distribution is Haar (uniform on the orthogonal group).

**Removing the correction breaks it.** `numpy.linalg.qr` returns an $R$ whose diagonal can have
either sign, and the convention it happens to use biases the result. The fix
`Q * np.sign(np.diag(R))` forces a canonical choice, which restores uniformity. Without it the
determinant distribution skews and, more subtly, so does the distribution of the columns.

### 4.2 How $\|Q^TQ - I\|$ grows

Fitting $\log\|Q^TQ - I\|$ against $\log n$ for `numpy.linalg.qr` gives an exponent near
$\tfrac12$, so the growth is $O(u\sqrt{n})$.

This is the expected result for a **backward stable** algorithm: Householder QR has an error
bound of $O(u\sqrt{n})$ or $O(un)$ depending on how carefully the constants are tracked, and
the observed $\sqrt{n}$ reflects statistical cancellation rather than the worst case.

### 4.3 The $1/\cos\theta$ law across its full range

Construct subspaces at controlled angle by rotating an orthonormal basis:

```python
def subspace_at_angle(m, k, theta, rng):
    Q = np.linalg.qr(rng.standard_normal((m, 2 * k)))[0]
    A, B_perp = Q[:, :k], Q[:, k:2*k]
    return A, A * np.cos(theta) + B_perp * np.sin(theta)
```

The law holds until $\cos\theta$ approaches $u$, at which point $Q_B^TQ_A$ is numerically
singular and the projector cannot be formed at all. That is around $\theta_{\max}$ within
$10^{-8}$ of $\pi/2$, so about 8 decades of the law are observable, which is what section 7
measured.

### 5.1 The CS decomposition

> **Theorem.** Let $Q$ be orthogonal, partitioned as
> $\left(\begin{smallmatrix}Q_{11} & Q_{12}\\ Q_{21} & Q_{22}\end{smallmatrix}\right)$ with
> $Q_{11}$ of size $p \times q$. Then there are orthogonal $U_1, U_2, V_1, V_2$ with
> $$\begin{pmatrix}U_1^T & \\ & U_2^T\end{pmatrix} Q \begin{pmatrix}V_1 & \\ & V_2\end{pmatrix}
> = \begin{pmatrix} C & -S \\ S & C\end{pmatrix}$$
> where $C$ and $S$ are diagonal with non-negative entries and $C^2 + S^2 = I$.

The diagonal entries of $C$ are $\cos\theta_i$ for the principal angles.

**Second proof of Theorem 16.10.** The block $Q_{11}$ is exactly $Q_B^TQ_A$ for suitable bases,
so its singular values are the $\cos\theta_i$, and $\|(Q_B^TQ_A)^{-1}\|_2 = 1/\min_i\cos\theta_i
= 1/\cos\theta_{\max}$, which is the claim. The CS decomposition is what makes "the singular
values are cosines" a theorem rather than a definition.

### 5.2 Projection in an $M$-inner product

For symmetric positive definite $M$, define $\langle x, y\rangle_M = x^TMy$ and $\|x\|_M =
\sqrt{x^TMx}$. This is a genuine inner product exactly because $M$ is positive definite
(lesson 20).

The $M$-orthogonal projector onto $\operatorname{range}(A)$ is

$$P_M = A(A^TMA)^{-1}A^TM.$$

It satisfies $P_M^2 = P_M$, and $MP_M = P_M^TM$, which is self-adjointness **in the $M$ inner
product** rather than the ordinary one. In the ordinary inner product it is oblique, and its
ordinary norm can be large; in the $M$ norm it is exactly 1.

**Conjugate gradient is this with $M = A$.** CG minimises $\|x - x_k\|_A$, and its iterates are
$A$-orthogonal projections onto Krylov subspaces. That is why CG requires $A$ symmetric positive
definite: without it, $\|\cdot\|_A$ is not a norm and the minimisation is meaningless. Lesson 24.

### 5.3 Orthogonality of $Q$ is not stability of the algorithm

The distinction is sharp and worth keeping.

- **If $Q$ is exactly orthogonal**, it cannot amplify error. That is Theorem 16.3, and it is a
  statement about the matrix.
- **The process that computed $Q$** can still be unstable, producing a $\tilde{Q}$ that is not
  orthogonal. Then Theorem 16.3 does not apply to $\tilde{Q}$, because its hypothesis is false.

Measure this by running classical Gram-Schmidt on an ill-conditioned matrix, getting
$\|\tilde{Q}^T\tilde{Q} - I\| \approx \kappa^2 u$, then checking whether $\tilde{Q}$ preserves
norms. It does not, and the failure is exactly the size of the orthogonality defect.

The correct summary: **"use orthogonal transformations" is advice about algorithm design, and it
only pays off if the transformations are computed stably.** Householder reflectors achieve that
(their orthogonality error is $O(u)$ regardless of $\kappa$); classical Gram-Schmidt does not.

---

## Lesson 17, Gaussian Elimination and LU

### 1.1 The saving at $n = 800$ with 500 right-hand sides

- Factor once, then 500 solves: $\tfrac{2}{3}n^3 + 500\cdot 2n^2$.
- Refactor each time: $500(\tfrac{2}{3}n^3 + 2n^2)$.

Measured with the exact flop counts at $n = 800$:

| | flops |
|---|---|
| factor once, solve 500 times | 981,652,400 |
| refactor 500 times | 171,466,200,000 |

**A saving of 99.43 percent**, a factor of 175.

### 1.2 Why $L$ is free

The multipliers $m_{ik} = a_{ik}/a_{kk}$ must be computed anyway, because elimination needs
them. $L$ is the matrix of those multipliers. Writing them down costs nothing beyond the
storage, and even that is free in practice because they go into the array positions that
elimination has just zeroed.

### 1.3 Twelve digits despite $\kappa = 10^{14}$

Because back substitution is **componentwise** backward stable (Theorem 17.4): the perturbation
satisfies $|\delta u_{ij}| \le nu|u_{ij}|$ entry by entry, not merely in norm.

The normwise bound $\kappa u \approx 10^{-2}$ allows a large error. The componentwise structure
is much stronger, and the relevant condition number is Skeel's (exercise 5.3), which can be far
smaller than $\kappa$. Section 7 measured the forward error 700 times better than $\kappa u$
allows at $n = 40$.

### 2.1 Unit lower triangular matrices form a group

**Product.** If $L, M$ are unit lower triangular then $(LM)_{ij} = \sum_k \ell_{ik}m_{kj}$. For
$i < j$, every term needs $k \le i < j \le k$, impossible, so the entry is 0. For $i = j$ the
only surviving term is $\ell_{ii}m_{ii} = 1$.

**Inverse.** $L$ is invertible since its determinant is 1. Solving $LX = I$ by forward
substitution produces an $X$ that is lower triangular with unit diagonal, by the same index
argument. $\square$

This is what makes Theorem 17.2 work: the product of all the elementary inverses stays in the
class.

### 2.2 Why the elementary inverses combine with no cross terms

$E_{ik} = I - m_{ik}\mathbf{e}_i\mathbf{e}_k^T$ has inverse $I + m_{ik}\mathbf{e}_i\mathbf{e}_k^T$,
since

$$(I - m\mathbf{e}_i\mathbf{e}_k^T)(I + m\mathbf{e}_i\mathbf{e}_k^T)
= I - m^2\mathbf{e}_i(\mathbf{e}_k^T\mathbf{e}_i)\mathbf{e}_k^T = I,$$

because $\mathbf{e}_k^T\mathbf{e}_i = 0$ for $i \ne k$, which holds since $i > k$.

Multiplying two such inverses:

$$(I + m_{ik}\mathbf{e}_i\mathbf{e}_k^T)(I + m_{jl}\mathbf{e}_j\mathbf{e}_l^T)
= I + m_{ik}\mathbf{e}_i\mathbf{e}_k^T + m_{jl}\mathbf{e}_j\mathbf{e}_l^T
+ m_{ik}m_{jl}\mathbf{e}_i(\mathbf{e}_k^T\mathbf{e}_j)\mathbf{e}_l^T.$$

The cross term needs $\mathbf{e}_k^T\mathbf{e}_j \ne 0$, that is $k = j$. But elimination
applies these in an order where $k < j$ always (column $k$ is finished before column $j$), so
$k = j$ never occurs among the terms being combined. **This is exactly where $i > k$ is
used.** $\square$

### 2.3 The exact LU flop count

At step $k$ there are $n - k - 1$ rows below the pivot. Each needs:

- one division for the multiplier: $n - k - 1$ divisions,
- $(n-k)$ multiply-add pairs to update the row: $2(n-k-1)(n-k)$ flops.

Summing over $k = 0, \dots, n-2$ and writing $j = n - k$:

$$\sum_{j=2}^{n}\big[(j-1) + 2(j-1)j\big]
= \sum_{j=2}^{n}(2j^2 - j - 1)
= \frac{2n^3}{3} - \frac{n^2}{2} - \frac{n}{6}. \qquad\square$$

Verified in section 2: the ratio to $\tfrac{2}{3}n^3$ is 1.0007 at $n = 1000$.

### 2.4 Existence and uniqueness of LU

**Existence.** Induct on $n$. Write $A$ in block form with $a_{11} \ne 0$ (the first leading
minor). One elimination step gives $A = \left(\begin{smallmatrix}1 & 0\\ \mathbf{l} &
I\end{smallmatrix}\right)\left(\begin{smallmatrix}a_{11} & \mathbf{u}^T\\ 0 &
S\end{smallmatrix}\right)$ where $S$ is the Schur complement. The leading minors of $S$ are
ratios of leading minors of $A$, so they are nonzero, and the induction applies.

**Uniqueness.** Suppose $L_1U_1 = L_2U_2$. Then $L_2^{-1}L_1 = U_2U_1^{-1}$. The left side is
unit lower triangular (exercise 2.1) and the right side is upper triangular. A matrix that is
both is diagonal, and unit lower triangular forces that diagonal to be $I$. So $L_1 = L_2$ and
$U_1 = U_2$. $\square$

### 2.5 Componentwise stability of back substitution, $2\times2$

Solve $\left(\begin{smallmatrix}u_{11} & u_{12}\\ 0 & u_{22}\end{smallmatrix}\right)\mathbf{x} =
\mathbf{y}$.

$\hat{x}_2 = \mathrm{fl}(y_2/u_{22}) = \dfrac{y_2}{u_{22}}(1 + \delta_1)$ with
$|\delta_1| \le u$. So $\hat{x}_2$ exactly solves $u_{22}(1+\delta_1)^{-1}\hat{x}_2 = y_2$,
which is a **relative** perturbation of $u_{22}$ alone.

$\hat{x}_1 = \mathrm{fl}\big((y_1 - \mathrm{fl}(u_{12}\hat{x}_2))/u_{11}\big)$. Each operation
contributes a factor $(1 + \delta)$, and collecting them gives

$$u_{11}(1+\epsilon_1)\hat{x}_1 + u_{12}(1+\epsilon_2)\hat{x}_2 = y_1$$

with $|\epsilon_i| \le 3u + O(u^2)$. Every perturbation multiplies its own entry, which is the
componentwise statement. $\square$

**The induction.** Row $i$ involves $n - i$ products and $n - i$ additions, so the accumulated
factor is $(1 + \delta)^{n-i+1}$, bounded by $1 + nu + O(u^2)$. That gives $|\delta u_{ij}| \le
nu|u_{ij}|$. Section 7 verified the componentwise backward error stays below $nu$ at every size.

### 3.1 In-place LU

```python
def lu_in_place(A):
    A = np.array(A, dtype=float)                 # one copy, then no more allocation
    n = A.shape[0]
    for k in range(n - 1):
        A[k+1:, k] /= A[k, k]                    # L below the diagonal
        A[k+1:, k+1:] -= np.outer(A[k+1:, k], A[k, k+1:])
    return A                                     # L is strict lower, U is upper
```

$L$'s unit diagonal is not stored, because it is known. The extra memory is $O(1)$ rather than
$O(n^2)$, which at $n = 10^4$ is the difference between 800 MB and 1.6 GB.

### 3.2 Blocked LU

Partition into $b \times b$ blocks and factor block-column by block-column. The updates become
matrix-matrix products, which are BLAS-3.

Measured speedups at $n = 2000$ are typically **5 to 20 times**, with an optimum $b$ around 64
to 256 depending on cache size. The flop count is identical; the gain is entirely lesson 08's
point that BLAS-3 reuses each loaded value $O(b)$ times while BLAS-2 uses it once.

### 3.3 Doolittle in direct form

```python
def doolittle_direct(A):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    L, U = np.eye(n), np.zeros((n, n))
    for i in range(n):
        U[i, i:] = A[i, i:] - L[i, :i] @ U[:i, i:]
        if i + 1 < n:
            L[i+1:, i] = (A[i+1:, i] - L[i+1:, :i] @ U[:i, i]) / U[i, i]
    return L, U
```

Agrees with elimination to roundoff, because it computes the same quantities in a different
order. It is the same relationship as Cholesky's direct derivation in lesson 20.

### 4.1 The empirical exponent of LU

Fitting a power law to timings from $n = 100$ to $2000$ gives an exponent **below 3**, typically
2.6 to 2.9, for a library implementation.

That is not a faster algorithm. It is lesson 08: at small $n$ the computation is memory bound
and the machine achieves a low fraction of peak; at larger $n$ blocking takes effect and the
achieved Gflop/s **rises**, so the measured time grows more slowly than $n^3$. The exponent
approaches 3 as $n$ grows past the point where peak throughput is reached.

### 4.2 Triangular forward error against $\kappa u$

The forward error sits **well below** the line $\kappa u$, by a factor of 100 to 1000 in the
measurements of section 7.

The gap does **not** shrink with $n$; if anything it grows, because the componentwise bound
becomes relatively stronger as more entries are involved. Exercise 5.3 identifies the sharper
bound that explains it.

### 4.3 `lu_factor` against `scipy.linalg.lu_factor`

The gap is a factor of $10^3$ to $10^4$, and it splits roughly as:

- **Python interpreter overhead**: the largest share at small $n$, since each inner operation
  is a NumPy call with fixed cost,
- **BLAS**: vectorised, multi-threaded kernels, the largest share at large $n$,
- **Blocking**: cache reuse, worth 5 to 20 times on its own.

None of it is algorithmic. The flop counts are identical.

### 5.1 Strassen

Strassen multiplies $2\times2$ block matrices with **7** multiplications instead of 8, giving
$O(n^{\log_2 7}) = O(n^{2.807})$.

The crossover against a tuned BLAS is typically $n \approx 1000$ or higher, and often never in
practice, for three reasons:

1. **A much larger constant** and far more additions.
2. **Worse memory behaviour**: the recursion needs temporaries, breaking the cache blocking that
   makes standard multiplication fast.
3. **Weaker stability.** Standard multiplication is componentwise backward stable; Strassen is
   only **normwise** stable, with error bounds involving $\|A\|\|B\|$ rather than $|A||B|$. For
   matrices with entries of widely varying size that is a real loss.

Point 3 is the one usually forgotten. A faster exponent that gives up a stability class is not a
free improvement.

### 5.2 LU as rank-one downdates

One elimination step computes

$$A^{(1)} = A - \mathbf{l}_1\mathbf{u}_1^T$$

on the trailing submatrix, where $\mathbf{l}_1$ is the first column of $L$ and $\mathbf{u}_1^T$
the first row of $U$. So

$$A = \sum_{k=1}^{n}\mathbf{l}_k\mathbf{u}_k^T,$$

which is lesson 15's outer product view of $LU$.

**The blocked algorithm falls out**: instead of removing one rank-one piece at a time, remove
$b$ of them at once as a single rank-$b$ update $L_{1:b}U_{1:b}$, which is one matrix-matrix
product. That is exactly exercise 3.2, derived rather than assembled.

### 5.3 The Skeel condition number

$$\operatorname{cond}(A, \mathbf{x}) = \frac{\big\||A^{-1}||A||\mathbf{x}|\big\|_\infty}{\|\mathbf{x}\|_\infty}.$$

It uses **absolute values entrywise**, so it measures sensitivity to componentwise relative
perturbations rather than normwise ones.

**Why it is smaller.** $\kappa_\infty(A) = \||A^{-1}||A|\|_\infty$ takes the worst $\mathbf{x}$;
Skeel's uses the actual $\mathbf{x}$. And it is invariant under **row scaling**, where $\kappa$
is not: multiplying a row by $10^{10}$ multiplies $\kappa$ by about $10^{10}$ and leaves Skeel's
unchanged.

**The sharper bound.** Combining with Theorem 17.4,

$$\frac{\|\hat{\mathbf{x}} - \mathbf{x}\|_\infty}{\|\mathbf{x}\|_\infty}
\le n u \operatorname{cond}(U, \mathbf{x}) + O(u^2),$$

and for the triangular matrices of section 7 the Skeel condition number is **orders of
magnitude below** $\kappa$, which is precisely the measured gap. The forward error was never
mysterious; it was being compared against the wrong condition number.

---

## Lesson 18, Pivoting and PA = LU

### 1.1 Why $|m_{ik}| \le 1$

The pivot is chosen as the largest-magnitude entry in the remaining part of the column, so
$|a_{kk}| \ge |a_{ik}|$ for every $i > k$. Hence $|m_{ik}| = |a_{ik}|/|a_{kk}| \le 1$. $\square$

### 1.2 Large backward error on a well conditioned matrix

**Pivot growth.** Everything else in Wilkinson's bound $\|\delta A\| \le cn^2\rho u\|A\|$ is
modest, so a large backward error implicates $\rho$. Section 4 measured exactly this on
Wilkinson's matrix: $\kappa \approx 27$, growth $5.8\times10^{17}$, every digit gone.

### 1.3 Permutation matrices are orthogonal

$P$ has exactly one 1 in each row and column and zeros elsewhere, so $P^TP$ has $(i,j)$ entry
equal to the inner product of columns $i$ and $j$ of $P$, which is 1 when $i = j$ and 0
otherwise. So $P^TP = I$.

Hence $\kappa_2(P) = 1$: permuting rows can never amplify error. That is why the permutation in
$PA = LU$ is numerically free as well as computationally free.

### 2.1 $\rho \le 2^{n-1}$ for partial pivoting

With partial pivoting $|m_{ik}| \le 1$, so at each elimination step

$$|a_{ij}^{(k+1)}| = |a_{ij}^{(k)} - m_{ik}a_{kj}^{(k)}|
\le |a_{ij}^{(k)}| + |a_{kj}^{(k)}| \le 2\max_{ij}|a_{ij}^{(k)}|.$$

So the largest entry at most **doubles** per step. There are $n-1$ steps, giving

$$\max|u_{ij}| \le 2^{n-1}\max|a_{ij}|, \qquad \rho \le 2^{n-1}. \qquad\square$$

### 2.2 Wilkinson's matrix attains it

For $W$ with 1 on the diagonal, $-1$ below, and a final column of ones: at each step the pivot
is the diagonal 1, which is already the largest in its column, so **no swap occurs**. The
multiplier is $-1$ for every row below.

Eliminating with multiplier $-1$ **adds** the pivot row to each row below. The last column
therefore doubles at every step: it starts as all ones and after step $k$ the entries below row
$k$ are $2^k$.

After $n-1$ steps the bottom right entry is $2^{n-1}$, so $\rho = 2^{n-1}$ exactly. Section 4
verified the ratio is 1.000000 at every size tested, and that the swap count is 0.

### 2.3 $P^{-1} = P^T$ and $\det P = (-1)^{\text{swaps}}$

The first is exercise 1.3. For the determinant, each row swap is an elementary operation with
determinant $-1$, and $P$ is the product of the swaps applied to $I$, so $\det P$ is $(-1)$ to
the number of swaps.

This is how `plu_factor` computes $\det A = (-1)^{\text{swaps}}\prod_i u_{ii}$, verified in
section 2 against `numpy.linalg.det`.

### 2.4 Column diagonal dominance means no swaps, and $\rho \le 2$

**No swaps.** Strict column diagonal dominance says $|a_{kk}| > \sum_{i\ne k}|a_{ik}|
\ge |a_{ik}|$ for every $i$, so the diagonal entry is already the largest in its column and
partial pivoting has nothing to swap.

**Preserved.** The Schur complement of a column diagonally dominant matrix is column diagonally
dominant, so the property holds at every step.

**Growth.** Column dominance gives $\sum_i|a_{ij}^{(k+1)}| \le \sum_i|a_{ij}^{(k)}|$ for each
column, so column sums do not grow, and since the diagonal dominates, $\rho \le 2$. $\square$

**Measured**: over 600 random column diagonally dominant matrices at $n = 5, 20, 60$, the total
number of swaps was **0** and the worst growth factor was **1.157**.

This is why tridiagonal solvers (lesson 21) skip pivoting: discretised differential operators
are diagonally dominant.

### 2.5 Wilkinson's backward error bound

Each elimination step introduces errors bounded by $u$ times the entries involved. Accumulating
over $n$ steps, the computed factors satisfy $\hat{L}\hat{U} = A + \delta A$ with

$$|\delta A| \le c\,n\,u\,|\hat{L}||\hat{U}|$$

componentwise. Since $|\hat{\ell}_{ij}| \le 1$ with partial pivoting and $\max|\hat{u}_{ij}| =
\rho\max|a_{ij}|$, taking the infinity norm gives

$$\|\delta A\|_\infty \le c\,n^2\,\rho\,u\,\|A\|_\infty. \qquad\square$$

The $n^2$ comes from the norm of $|\hat{L}||\hat{U}|$: $n$ from the matrix product and $n$ from
the accumulation.

### 3.1 In-place partial pivoting with an index array

Swap **index entries** rather than moving rows, then apply the permutation once at the end if a
contiguous layout is needed. This avoids $O(n^2)$ data movement per swap.

Measured at $n = 1000$, the index version is 2 to 5 times faster than the row-moving version,
and the gap widens with $n$ because memory traffic dominates.

### 3.2 Rook pivoting

Alternate searching the column and the row until an entry is largest in **both**. It costs
$O(n^2)$ on average like partial pivoting, and it has far better growth behaviour, with a bound
of $1.5\,n^{0.75\log n}$.

On Wilkinson's matrix rook pivoting **finds the problem and avoids it**, giving growth near 1
rather than $2^{n-1}$. On random matrices it is indistinguishable from partial pivoting. It is a
reasonable default that history simply did not choose.

### 3.3 `solve_and_check`

```python
def solve_and_check(A, b):
    fac = pv.plu_factor(A)
    x = pv.plu_solve(fac, b)
    d = ls.diagnose_system(A, b, x)
    d["growth"] = fac["growth"]
    return x, d
```

On the four matrices of section 6 it flags exactly the Wilkinson rows, because they are the only
ones with a large **backward** error. The Hilbert and beam matrices have perfect backward errors
and hard problems, which is a different diagnosis entirely.

### 4.1 Growth for different random families

Gaussian, uniform on $[0,1]$ and uniform on $[-1,1]$ all give small growth, a few units. The
**uniform on $[0,1]$** family is slightly worse, because all entries share a sign so there is
less cancellation to keep entries small.

**Orthogonal** matrices give growth very close to 1, because their entries are bounded by 1 and
elimination cannot make them grow much.

### 4.2 Searching for large growth at $n = 10$

Local search from random starts reaches growth of about 20 to 60 against the bound $2^9 = 512$.
Reaching the bound requires the exact sign pattern of Wilkinson's matrix, and random search
essentially never finds it, which is itself evidence for why large growth does not occur in
practice.

### 4.3 The mean rather than the median

Fitting the **mean** growth over many trials up to $n = 256$ gives an exponent closer to $2/3$
than the median's $0.54$.

The mean is larger and grows faster because the distribution has a right tail: occasional
matrices have noticeably larger growth, and they pull the mean up while leaving the median
alone. Trefethen and Bau's $n^{2/3}$ conjecture is about the average, so this is the right
statistic to compare against, and lesson 18 says so rather than quoting the folklore figure for
its median measurement.

### 5.1 Why large growth is so rare

The leading explanation is that large growth requires the multipliers to align in sign across
many steps, so that updates **add** rather than cancel. In Wilkinson's matrix every multiplier is
exactly $-1$ and every update adds. A random matrix produces multipliers of mixed sign, and the
updates cancel on average.

A good experiment: take Wilkinson's matrix and randomly flip the sign of a fraction $p$ of its
subdiagonal entries. Growth should collapse rapidly as $p$ rises from 0, showing the alignment
is fragile.

### 5.2 Threshold pivoting

Pivot only when the candidate is smaller than $\tau$ times the column maximum, with
$0 < \tau \le 1$. Then $|m_{ik}| \le 1/\tau$, so growth is bounded by $(1 + 1/\tau)^{n-1}$,
worse than partial pivoting but still bounded.

The trade-off curve is monotone: as $\tau$ falls from 1 to 0, fill decreases and growth
increases. Sparse solvers typically use $\tau \approx 0.1$, accepting a factor of 10 in the
multiplier bound to gain a large reduction in fill.

### 5.3 Growth in Cholesky and QR

**Cholesky has $\rho = 1$** because $|\ell_{ij}| \le \sqrt{a_{ii}}$ (lesson 20, exercise 2.2),
so no entry can exceed the square root of the largest diagonal. Positive definiteness is
inherited, so there is never a bad pivot.

**QR has no growth at all** because it uses orthogonal transformations, and lesson 16 proved
those cannot amplify anything: $\|Q^TA\|_2 = \|A\|_2$ exactly.

**When to prefer QR for a square system.** QR costs $\tfrac{4}{3}n^3$, twice LU. Pay that when:

- the matrix is ill conditioned **and** you cannot verify the residual cheaply,
- pivot growth is a genuine concern for the structure at hand,
- you need the factorization for least squares anyway.

Otherwise LU with partial pivoting plus a residual check (section 6) costs half as much and
catches the rare failure when it happens. That is the trade every library has made.

---

## Lesson 19, Conditioning of Linear Systems

### 1.1 Residual $10^{-16}$ with $\kappa = 10^{12}$

The bound is $\kappa \times \text{backward error} = 10^{12}\times10^{-16} = 10^{-4}$.

So about **4 correct digits**. The residual alone suggested 16. The product is what matters.

### 1.2 Scaling does not change $\kappa$

$\kappa(cA) = \|cA\|\|(cA)^{-1}\| = |c|\|A\| \cdot |c|^{-1}\|A^{-1}\| = \kappa(A)$.

The scale cancels, which is exactly the invariance that makes $\kappa$ meaningful and the raw
residual meaningless. Section 1 measured the raw residual moving 16 orders of magnitude under
scaling while the scaled one did not move.

### 1.3 Same $\kappa$, different difficulty

**No.** $\kappa$ is a worst case over right-hand sides. A given $\mathbf{b}$ may excite the worst
direction for one matrix and not the other.

Section 4 measured this: over 3000 right-hand sides the **median** magnification was orders of
magnitude below $\kappa$, and only the maximum approached it. Two matrices with the same
$\kappa$ can behave very differently on the same $\mathbf{b}$.

### 2.1 The bound in Theorem 19.1 is attained

Use the SVD $A = U\Sigma V^T$. Choose

$$\mathbf{b} = \mathbf{u}_1\sigma_1 \ \text{(so } \mathbf{x} = \mathbf{v}_1),
\qquad \delta\mathbf{b} = \epsilon\,\mathbf{u}_n.$$

Then $\delta\mathbf{x} = A^{-1}\delta\mathbf{b} = \epsilon\mathbf{v}_n/\sigma_n$, giving

$$\frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|} = \frac{\epsilon}{\sigma_n},
\qquad \frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|} = \frac{\epsilon}{\sigma_1},$$

so the ratio is exactly $\sigma_1/\sigma_n = \kappa_2(A)$. $\square$

The bound is sharp, and attaining it requires $\mathbf{b}$ along the **largest** singular
direction and $\delta\mathbf{b}$ along the **smallest**. That specific alignment is why random
perturbations fall so far short.

### 2.2 The version with $\|\mathbf{x}\|$ in the denominator

From $A(\hat{\mathbf{x}} - \mathbf{x}) = -\delta A\hat{\mathbf{x}}$,

$$\|\hat{\mathbf{x}} - \mathbf{x}\| \le \|A^{-1}\|\|\delta A\|\|\hat{\mathbf{x}}\|
\le \|A^{-1}\|\|\delta A\|(\|\mathbf{x}\| + \|\hat{\mathbf{x}} - \mathbf{x}\|).$$

Let $t = \|\hat{\mathbf{x}} - \mathbf{x}\|/\|\mathbf{x}\|$ and $\gamma = \|A^{-1}\|\|\delta A\|$.
Then $t \le \gamma(1 + t)$, so $t(1 - \gamma) \le \gamma$ and

$$\frac{\|\hat{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|}
\le \frac{\kappa(A)\|\delta A\|/\|A\|}{1 - \kappa(A)\|\delta A\|/\|A\|}. \qquad\square$$

The extra factor is close to 1 exactly when $\gamma < 1$, which is the hypothesis of the
theorem.

### 2.3 Perturbing both at once

With $(A + \delta A)(\mathbf{x} + \delta\mathbf{x}) = \mathbf{b} + \delta\mathbf{b}$ and
$\gamma = \kappa\|\delta A\|/\|A\| < 1$,

$$\frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|}
\le \frac{\kappa}{1 - \gamma}\left(\frac{\|\delta A\|}{\|A\|}
+ \frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}\right).$$

The two sources simply add, each amplified by $\kappa$. The derivation combines 19.1 and 19.2
with the same algebra as exercise 2.2.

### 2.4 $\kappa_2(A) = \sigma_1/\sigma_n$

$\|A\|_2 = \sigma_1$ by definition of the largest singular value. For the inverse, if
$A = U\Sigma V^T$ then $A^{-1} = V\Sigma^{-1}U^T$, whose singular values are $1/\sigma_i$. Its
largest is $1/\sigma_n$. So

$$\kappa_2(A) = \sigma_1 \cdot \frac{1}{\sigma_n} = \frac{\sigma_1}{\sigma_n}. \qquad\square$$

Geometrically (lesson 15 section 4): $A$ maps the unit sphere to an ellipsoid with semi-axes
$\sigma_i$, and $\kappa_2$ is how squashed that ellipsoid is.

### 2.5 $\kappa \ge 1$, with equality for multiples of orthogonal matrices

$\kappa \ge 1$ is lesson 16 exercise 1.1.

**Equality in the 2-norm** requires $\sigma_1 = \sigma_n$, so all singular values are equal, say
to $c$. Then $A = U(cI)V^T = c(UV^T)$, a scalar multiple of an orthogonal matrix. Conversely any
such matrix has all singular values equal to $|c|$ and $\kappa_2 = 1$. $\square$

So the perfectly conditioned matrices are exactly the scaled orthogonal ones, which is another
way of saying lesson 16's point.

### 3.1 `solve_and_report`

Combine `nalib.refinement.condition_estimate` (lesson 22, $O(n^2)$) with
`nalib.linsys.diagnose_system`:

```python
def solve_and_report(A, b):
    fac = pv.plu_factor(A)
    x = pv.plu_solve(fac, b)
    report = ls.diagnose_system(A, b, x)
    report["kappa_estimate"] = rf.condition_estimate(A, fac)   # O(n^2)
    report["growth"] = fac["growth"]
    return x, report
```

On section 7's four matrices it produces four distinct verdicts, which is the point: the two
failure modes look identical from the residual alone.

### 3.2 The Hilbert condition number computed exactly

With `fractions.Fraction` the Hilbert matrix is exact, and its inverse has known integer
entries, so $\kappa$ can be computed with no rounding at all.

Comparing against the floating point value: they agree until about $n = 12$, after which the
**floating point $\kappa$ is itself unreliable**, because computing it requires inverting a
matrix that is numerically singular. Beyond $n \approx 13$ the reported $\kappa$ saturates near
$1/u$ regardless of the true value, which is why lesson 19's table stops rising past $10^{18}$.

That is a good lesson in itself: the diagnostic has its own accuracy limit.

### 3.3 The Skeel condition number

$$\operatorname{cond}(A, \mathbf{x}) = \frac{\||A^{-1}||A||\mathbf{x}|\|_\infty}{\|\mathbf{x}\|_\infty}.$$

It is **invariant under row scaling** where $\kappa$ is not. Construct a matrix, multiply one
row by $10^{10}$, and $\kappa$ jumps by about $10^{10}$ while Skeel's is unchanged.

For badly scaled matrices the two can differ by many orders of magnitude, and Skeel's gives the
sharper and more honest forward error bound. See lesson 17 exercise 5.3.

### 4.1 Magnification against dimension

For fixed $\kappa$, a random right-hand side gets **relatively further** from the worst case as
$n$ grows, because attaining $\kappa$ needs alignment with a specific singular direction and
random vectors in high dimensions are nearly orthogonal to any fixed direction.

The typical magnification scales like $\kappa/\sqrt{n}$ rather than $\kappa$. This is the same
concentration effect as lesson 15's failed norm sampling.

### 4.2 The true beam trade-off

Replace the representative $1/n^2$ with the measured error against the analytic beam solution.
The optimum shifts, but the **shape** is unchanged: a U-shaped total error with a definite
minimum.

The lesson's optimum at $n = 256$ is representative rather than exact, and the lesson says so by
labelling the column as a model error.

### 4.3 $\kappa_1$, $\kappa_2$, $\kappa_\infty$ compared

They satisfy $\kappa_2 \le \sqrt{\kappa_1\kappa_\infty}$ and are within a factor of $n$ of each
other. For most matrices they agree to within a small factor.

**It essentially never changes a decision**, because the decision is made on the order of
magnitude. That is why LAPACK estimates $\kappa_1$ (cheap) rather than $\kappa_2$ (an SVD).

### 5.1 Componentwise conditioning

Define $\operatorname{cond}(A) = \||A^{-1}||A|\|_\infty$. Under row scaling $A \to DA$:

$$|(DA)^{-1}||DA| = |A^{-1}D^{-1}||DA| = |A^{-1}||A|,$$

since $D$ is diagonal and the absolute values cancel. So it is **exactly invariant**, while
$\kappa(DA)$ can be made arbitrarily large by choosing $D$.

A matrix with rows of wildly different scale, such as $\operatorname{diag}(1, 10^{10})A$, shows
a gap of $10^{10}$ between the two.

### 5.2 Equilibration

Scale rows and columns so all have comparable norms. The van der Sluis theorem says row
equilibration brings $\kappa_\infty$ within a factor of $n$ of its minimum over all row
scalings.

**Why LAPACK does not do it by default.** Scaling changes the problem being solved: it changes
which componentwise errors are considered small. If the row scales carry physical meaning, for
instance different units, equilibrating discards that information. LAPACK provides `geequ` and
lets the caller decide, which is the right division of responsibility.

### 5.3 Distance to singularity

> **Theorem.** $\min\{\|\delta A\|_2/\|A\|_2 : A + \delta A \text{ singular}\} = 1/\kappa_2(A)$.

*Proof.* Write $A = U\Sigma V^T$ and take $\delta A = -\sigma_n\mathbf{u}_n\mathbf{v}_n^T$. Then
$A + \delta A$ has singular values $\sigma_1, \dots, \sigma_{n-1}, 0$, so it is singular, and
$\|\delta A\|_2 = \sigma_n$, giving a relative distance $\sigma_n/\sigma_1 = 1/\kappa_2$.

For the lower bound, if $A + \delta A$ is singular there is a unit $\mathbf{z}$ with
$(A + \delta A)\mathbf{z} = 0$, so $\|A\mathbf{z}\| = \|\delta A\mathbf{z}\| \le \|\delta A\|$.
But $\|A\mathbf{z}\| \ge \sigma_n$ for a unit vector, so $\|\delta A\| \ge \sigma_n$. $\square$

**This is the sharpest interpretation of $\kappa$**: it says a large condition number means
precisely that $A$ is *close to singular* in a relative sense. A matrix with
$\kappa = 10^{16}$ is within $10^{-16}$ of being singular, which is within rounding of the
entries themselves.

---

## Lesson 20, Symmetric Positive Definite and Cholesky

### 1.1 No zero on the diagonal

Take $\mathbf{x} = \mathbf{e}_i$. Then $\mathbf{e}_i^TA\mathbf{e}_i = a_{ii} > 0$ by the
definition. So every diagonal entry is strictly positive. $\square$

### 1.2 A negative diagonal entry

By exercise 1.1, the matrix is **not positive definite**, immediately and with no computation.
This is the cheapest possible test and it should be the first one applied.

### 1.3 Half the memory as well as half the flops

Symmetry means the strict upper triangle carries no information the lower triangle does not, so
only $n(n+1)/2$ entries need storing rather than $n^2$. And the Cholesky factor $L$ is
triangular, so it fits in the same space.

For large $n$ the memory saving is often the binding constraint rather than the flop saving.

### 2.1 The Schur complement of a positive definite matrix is positive definite

Partition $A = \left(\begin{smallmatrix}a & \mathbf{w}^T\\ \mathbf{w} & B\end{smallmatrix}\right)$
with $a > 0$. The Schur complement is $S = B - \mathbf{w}\mathbf{w}^T/a$.

For any $\mathbf{y} \ne 0$, choose $\mathbf{x} = (-\mathbf{w}^T\mathbf{y}/a, \mathbf{y})$. Then

$$0 < \mathbf{x}^TA\mathbf{x} = \mathbf{y}^TB\mathbf{y}
- \frac{(\mathbf{w}^T\mathbf{y})^2}{a} = \mathbf{y}^TS\mathbf{y}.$$

So $S$ is positive definite. $\square$

By induction Cholesky never meets a non-positive pivot, which proves the converse of Theorem
20.2 and hence Theorem 20.3's claim that no pivoting is needed.

### 2.2 $|\ell_{ij}| \le \sqrt{a_{ii}}$ and growth exactly 1

Row $i$ of $A = LL^T$ gives $a_{ii} = \sum_k \ell_{ik}^2$. Every term is non-negative, so each
individually satisfies $\ell_{ij}^2 \le a_{ii}$, that is $|\ell_{ij}| \le \sqrt{a_{ii}}$.
$\square$

So $\max|\ell_{ij}| \le \sqrt{\max_i a_{ii}} \le \sqrt{\max_{ij}|a_{ij}|}$, using exercise 2.5.
The entries of the factor cannot exceed the square root of the largest entry of $A$, so there is
no growth. Section 3 verified the bound at every size tested.

### 2.3 The Cholesky flop count

At step $j$: one square root, $j$ multiply-adds for the diagonal, and for each of the $n-j-1$
entries below, $j$ multiply-adds plus a division. Summing:

$$\sum_{j=0}^{n-1}\Big[(2j+1) + (n-j-1)(2j+1)\Big] = \frac{n^3}{3} + O(n^2). \qquad\square$$

Verified in section 4: the ratio to LU converges to exactly 0.5, and the count to $n^3/3$.

### 2.4 Sylvester's law of inertia

> If $M$ is nonsingular then $A$ and $MAM^T$ have the same inertia.

*Proof sketch.* Let $V_+$ be a maximal subspace on which $\mathbf{x}^TA\mathbf{x} > 0$, of
dimension $p$. Then $M^{-T}V_+$ has the same dimension and

$$\mathbf{y}^T(MAM^T)\mathbf{y} = (M^T\mathbf{y})^TA(M^T\mathbf{y}) > 0$$

on it, so $MAM^T$ has at least $p$ positive eigenvalues. Applying the same argument to $M^{-1}$
gives the reverse inequality, so the counts are equal. The same works for the negative and zero
counts. $\square$

Since $A = LDL^T$ is a congruence with $M = L^{-1}$, the signs of $D$ give the inertia of $A$
directly. Section 6 verified this against `eigvalsh` on definite, negative definite and
indefinite examples.

### 2.5 $a_{ij}^2 < a_{ii}a_{jj}$ for $i \ne j$

Take $\mathbf{x} = t\mathbf{e}_i + \mathbf{e}_j$. Then

$$0 < \mathbf{x}^TA\mathbf{x} = t^2a_{ii} + 2ta_{ij} + a_{jj}$$

for every real $t$. A quadratic in $t$ that is strictly positive everywhere has negative
discriminant:

$$4a_{ij}^2 - 4a_{ii}a_{jj} < 0 \implies a_{ij}^2 < a_{ii}a_{jj}. \qquad\square$$

**Consequence.** The largest entry of a positive definite matrix always lies on its diagonal,
since $|a_{ij}| < \sqrt{a_{ii}a_{jj}} \le \max(a_{ii}, a_{jj})$.

Measured: over 600 random SPD matrices at $n = 3, 10, 40$, there were **zero** violations.

### 3.1 In-place Cholesky

Overwrite the lower triangle of $A$ with $L$. The upper triangle is never read after
initialisation, since symmetry means it is redundant.

```python
def cholesky_in_place(A):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    for j in range(n):
        A[j, j] = np.sqrt(A[j, j] - A[j, :j] @ A[j, :j])
        if j + 1 < n:
            A[j+1:, j] = (A[j+1:, j] - A[j+1:, :j] @ A[j, :j]) / A[j, j]
    return np.tril(A)
```

### 3.2 Pivoted Cholesky for semi-definite matrices

At each step move the largest remaining diagonal entry to the pivot position, and stop when it
falls below a tolerance. The result is $P^TAP = LL^T$ with $L$ of size $n \times r$, revealing
the numerical rank $r$.

On a deliberately rank-deficient matrix the pivots drop sharply at exactly the rank, which
makes this a practical rank-revealing factorization at $O(n^2r)$ rather than the $O(n^3)$ an SVD
would cost.

### 3.3 Rank-one update of a Cholesky factor

Updating $A + \mathbf{v}\mathbf{v}^T$ can be done in $O(n^2)$ by applying $n$ Givens rotations to
$L$ augmented with $\mathbf{v}$, rather than refactorizing at $O(n^3)$.

At $n = 1000$ that is a factor of about 300. This is what makes Kalman filtering and sequential
least squares affordable, since both add one observation at a time.

### 4.1 A fair timing comparison

Section 4's timing was deliberately unfair, and the lesson says so: it compared LAPACK's
Cholesky against a Python LU. Implementing both in the same style gives a ratio near the
predicted **0.5**, approached from above at small $n$ where $O(n^2)$ terms still matter.

### 4.2 Where Cholesky fails through roundoff

A computed pivot goes negative when the accumulated rounding exceeds the true pivot, which
happens around $\kappa \approx 1/u \approx 10^{16}$.

Slightly before that, Cholesky may succeed while producing a factor with no accuracy. So a
successful Cholesky is evidence of positive definiteness **to working precision**, not a proof
about the exact matrix. That distinction matters when the matrix comes from data.

### 4.3 Inertia accuracy near zero

The sign of a $d_i$ becomes unreliable when $|d_i|$ falls to about $u\|A\|$, since the
accumulated rounding is of that size.

So the inertia via LDL$^T$ is reliable exactly when no eigenvalue is closer to zero than
$u\|A\|$, which is the same condition under which the eigenvalues themselves have a determinate
sign. Neither method can do better, because the sign is not determined by the stored data.

### 5.1 The block derivation of Cholesky

Partition and match:

$$\begin{pmatrix} a & \mathbf{w}^T \\ \mathbf{w} & B\end{pmatrix}
= \begin{pmatrix}\ell & \mathbf{0}^T \\ \mathbf{m} & L_{22}\end{pmatrix}
\begin{pmatrix}\ell & \mathbf{m}^T \\ \mathbf{0} & L_{22}^T\end{pmatrix}$$

gives $\ell = \sqrt{a}$, $\mathbf{m} = \mathbf{w}/\ell$, and
$L_{22}L_{22}^T = B - \mathbf{m}\mathbf{m}^T = B - \mathbf{w}\mathbf{w}^T/a$, the Schur
complement.

By exercise 2.1 that is positive definite, so the recursion is well defined. **Replacing the
scalar $a$ by a $b\times b$ block gives the blocked algorithm directly**, exactly as lesson 17
exercise 5.2 did for LU.

### 5.2 Bunch-Kaufman pivoting

Compute $PAP^T = LDL^T$ where $D$ has $1\times1$ and $2\times2$ diagonal blocks, choosing the
block size by a threshold test on the largest diagonal and largest off-diagonal entries.

**Why $2\times2$ blocks are unavoidable.** The matrix
$\left(\begin{smallmatrix}0&1\\1&0\end{smallmatrix}\right)$ is symmetric, nonsingular, and has
**no** $LDL^T$ factorization with diagonal $D$ and any symmetric permutation: every diagonal
entry is zero, so no $1\times1$ pivot exists. Taking the whole matrix as a single $2\times2$
block is the only option. Section 6 measured `nalib.cholesky.ldl` failing on exactly this
matrix, and `inertia` correctly falling back.

### 5.3 What conjugate gradient needs from this class

- **Symmetry** makes the Krylov recursion **three-term** rather than full, which is why CG
  stores three vectors instead of all previous ones. Without it you get GMRES (lesson 27), whose
  cost grows with the iteration count.
- **Positive definiteness** makes $\|\mathbf{x}\|_A = \sqrt{\mathbf{x}^TA\mathbf{x}}$ a genuine
  norm, and CG minimises the error in it. Without positive definiteness the quantity can be zero
  or negative for nonzero vectors, and "minimise" means nothing.

Drop symmetry and CG's short recurrence is invalid. Drop definiteness and its optimality
statement is meaningless. Both are essential, and lesson 24 makes each precise.

---

## Lesson 21, Banded, Sparse and Structured Systems

### 1.1 Memory at $n = 10^6$ tridiagonal

- **Dense**: $n^2 \times 8$ bytes $= 8\times10^{12}$ bytes, **8 terabytes**.
- **CSR**: $\text{nnz} \approx 3n$, so $2(3n) + n + 1 \approx 7n$ numbers $= 5.6\times10^7$
  bytes, about **56 megabytes**.

A factor of about 140,000. The dense version is impossible on any machine; the sparse version
fits comfortably in memory.

### 1.2 Why reordering changes no answer

Reordering is a **relabelling of the unknowns**: solving $(P A P^T)(P\mathbf{x}) = P\mathbf{b}$
gives the same $\mathbf{x}$ after undoing the permutation. Since $P$ is a permutation matrix it
is orthogonal, so the transformation is exact and carries no numerical cost either.

It changes only the **fill**, hence the work and memory of the factorization.

### 1.3 99 percent zero, cheap factorization?

**No.** Section 5 measured the arrow matrix at 90 percent zero factorizing to **completely
dense**. Sparsity of $A$ says nothing about sparsity of $L$ and $U$; the fill depends on the
ordering, and a bad ordering destroys the advantage entirely.

### 2.1 Banded elimination stays in the band

Row $i$ has nonzeros only in columns $[i-p, i+q]$. Elimination subtracts a multiple of row $k$
from row $i$, and this only happens when $a_{ik} \ne 0$, that is $i - k \le p$.

Row $k$ has nonzeros in $[k-p, k+q] \subseteq [i-2p, i+q]$. But the elimination only modifies
columns from $k$ onward, and $k \ge i-p$, so the modified entries lie in $[i-p, i+q]$, which is
row $i$'s own band. Nothing outside is touched. $\square$

**Why pivoting breaks it.** A row swap can bring a row whose nonzeros start much earlier, so the
lower bandwidth of the result can exceed $p$. LAPACK's banded solver reserves $p$ extra
superdiagonals in advance to hold exactly that fill.

### 2.2 The Thomas flop count

Forward sweep: at each of $n-1$ steps, one multiply, one subtract, one divide for $c'$, and
three more for $d'$. Back substitution: one multiply and one subtract per step.

Total is about $8n$ flops, which is $O(n)$. Against $\tfrac{2}{3}n^3$ for dense LU, at
$n = 10^4$ that is $8\times10^4$ against $6.7\times10^{11}$, a factor of eight million.

### 2.3 Diagonal dominance means no pivoting, growth $\le 2$

For a tridiagonal matrix with $|b_i| > |a_i| + |c_i|$: after eliminating, the new diagonal is
$b_i' = b_i - a_ic_{i-1}/b_{i-1}'$. Induction shows $|b_i'| > |c_i|$ at every step, so the
diagonal remains the largest entry in its column and no swap is needed.

For the growth bound, $|b_i'| \le |b_i| + |a_i||c_{i-1}|/|b_{i-1}'| \le |b_i| + |a_i|$, and
diagonal dominance bounds this by $2|b_i|$. $\square$

### 2.4 The second difference eigenvalues, and $\kappa \sim n^2$

Try $v_k = \sin(kj\pi/(n+1))$. Then

$$-v_{j-1} + 2v_j - v_{j+1} = \left(2 - 2\cos\frac{k\pi}{n+1}\right)v_j
= 4\sin^2\!\frac{k\pi}{2(n+1)}\,v_j,$$

using $1 - \cos\theta = 2\sin^2(\theta/2)$. The boundary conditions are satisfied because
$\sin 0 = \sin(k\pi) = 0$. $\square$

**The condition number.** Largest eigenvalue at $k = n$ is near 4; smallest at $k = 1$ is
$4\sin^2(\pi/(2(n+1))) \approx \pi^2/(n+1)^2$. So

$$\kappa_2 \approx \frac{4(n+1)^2}{\pi^2} \sim n^2.$$

**Measured**: fitting $\log\kappa$ against $\log n$ for $n = 8$ to $128$ gives an exponent of
**1.930**, converging to 2.

### 2.5 The inverse of a banded matrix is dense

The inverse of an irreducible banded matrix is **completely dense** in general.

*Reason.* Column $j$ of $A^{-1}$ solves $A\mathbf{x} = \mathbf{e}_j$, which is a discrete Green's
function. For the second difference operator this is a piecewise linear "tent" that is nonzero
almost everywhere, because a local disturbance in a differential equation has a **global**
effect.

**Measured**: the tridiagonal second difference matrix at $n = 12$ has bandwidth $(1,1)$ and its
inverse has density **1.000**, meaning every single entry is nonzero.

This is why you never form the inverse of a sparse matrix: you would turn $O(n)$ storage into
$O(n^2)$.

### 3.1 Banded storage

Store only the $p + q + 1$ diagonals in a $(p+q+1) \times n$ array, with LAPACK's convention that
$A[i,j]$ lives at `band[q + i - j, j]`.

Memory drops from $n^2$ to $n(p+q+1)$: at $n = 10^4$ tridiagonal that is $3\times10^4$ against
$10^8$.

### 3.2 Banded Cholesky

Combining both structures gives $O(np^2/2)$ flops and $O(np)$ storage, half of banded LU and
with no pivoting needed since positive definiteness rules it out (lesson 20).

**This is the best case in all of Part 3**, and it is exactly what a discretised self-adjoint
differential operator produces.

### 3.3 Minimum degree ordering

Greedily eliminate the vertex of lowest degree in the sparsity graph, updating degrees as fill
is created.

Compared against RCM: minimum degree usually gives **less fill**, and RCM usually gives **smaller
bandwidth**. They optimise different things. Modern solvers use approximate minimum degree
(AMD), which is minimum degree with cheaper degree estimates.

### 4.1 Fill for the 2D Laplacian

On an $m \times m$ grid with $n = m^2$ unknowns:

| Ordering | Fill |
|---|---|
| natural (row by row) | $O(n^{3/2})$ |
| RCM | $O(n^{3/2})$, better constant |
| nested dissection | $O(n\log n)$, and provably optimal |

Nested dissection is asymptotically better and is what modern sparse solvers use. In 3D the
gaps widen further, which is where the choice really bites.

### 4.2 The sparse-dense crossover

Much larger than the flop counts suggest, typically $n$ in the thousands, because sparse code
has **indirect addressing**: every operation follows an index array, defeating prefetching and
vectorisation.

Dense code at 10 percent of peak can beat sparse code at 1 percent of peak even with 10 times
the flops. This is lesson 08's point once more.

### 4.3 Adding off-band entries

The transition is **sudden**, not gradual. A tridiagonal matrix with a handful of random
off-band entries can already produce substantial fill, because each such entry creates a long
range coupling that propagates through elimination.

This is why "mostly banded" is not a useful category, and why sparse solvers reorder rather than
hoping.

### 5.1 Nested dissection

Split the graph with a separator $S$ whose removal disconnects it into $A$ and $B$. Order $A$,
then $B$, then $S$ last. Eliminating $A$ cannot create fill into $B$, because they are not
connected. Recurse.

For a 2D $m\times m$ grid the separator has $O(m) = O(\sqrt{n})$ vertices, giving fill
$O(n\log n)$ and work $O(n^{3/2})$, which George proved is optimal.

Confirming the exponent numerically requires care, since the constants are large and the
asymptotics take hold slowly.

### 5.2 Cyclic reduction

Eliminate every odd-indexed unknown simultaneously, leaving a tridiagonal system of half the
size in the even ones. Recurse: $\log_2 n$ levels.

**Depth $O(\log n)$** against Thomas's $O(n)$, at the cost of about twice the total work.

**When to trade work for depth.** On a parallel machine with $p$ processors, Thomas is entirely
sequential and gains nothing from $p$, while cyclic reduction achieves $O(\log n)$ depth. On a
single core Thomas wins on total work. The choice is made by the hardware, not the mathematics.

### 5.3 Toeplitz and the broader idea of structure

A Toeplitz matrix has constant diagonals, so it is **dense** but determined by only $2n-1$
numbers.

**Levinson-Durbin** exploits this to solve in $O(n^2)$ rather than $O(n^3)$, by building the
solution for leading submatrices of growing size and using the shift structure to update
cheaply. Superfast methods reach $O(n\log^2 n)$ via the FFT (Part 8).

**Structure is broader than sparsity.** A matrix can have $n^2$ nonzeros and still be described
by $O(n)$ parameters, and any such description can in principle be exploited. Other examples:
circulant matrices (diagonalised exactly by the FFT), Hankel, Vandermonde, and hierarchical
matrices where off-diagonal blocks are low rank. Sparsity is the special case where the
structure happens to be "most entries are zero".

---

## Lesson 22, Condition Estimation and Iterative Refinement

### 1.1 Why $\|A\|_1$ is free and $\|A^{-1}\|_1$ is not

$\|A\|_1$ is the maximum absolute column sum, computable by inspecting the $n^2$ entries once.
$\|A^{-1}\|_1$ needs $A^{-1}$, whose computation is $O(n^3)$, and which lesson 17 measured being
both slower and less accurate than solving.

### 1.2 An estimate of $10^7$ when the truth is $3\times10^7$

**No decision changes.** The question a condition number answers is "how many digits can I
trust", and $\log_{10}$ of the two differs by 0.5. You would conclude "about 9 correct digits"
either way.

That is exactly why an $O(n^2)$ estimate is worth as much as an $O(n^3)$ exact value.

### 1.3 Why the correction solve reuses the factorization

Because $A$ has not changed. The factorization $PA = LU$ is a property of $A$ alone, and the
correction solves $A\mathbf{d} = \mathbf{r}$, the **same** matrix with a different right-hand
side. Refactorizing would cost $O(n^3)$ to recompute something already known, making refinement
as expensive as solving again, which would destroy the point.

### 2.1 $\|A^{-1}\mathbf{x}\|_1$ is convex and maximised at a vertex

$f(\mathbf{x}) = \|A^{-1}\mathbf{x}\|_1$ is a norm composed with a linear map, so it is convex:

$$f(t\mathbf{x} + (1-t)\mathbf{y}) = \|A^{-1}(t\mathbf{x} + (1-t)\mathbf{y})\|_1
\le t f(\mathbf{x}) + (1-t)f(\mathbf{y}).$$

A convex function on a compact convex set attains its maximum at an **extreme point**. The unit
$1$-norm ball is the cross-polytope, whose extreme points are exactly $\pm\mathbf{e}_j$. So the
search is over $2n$ points. $\square$

### 2.2 Hager's method increases at every step, so it terminates

At each step the algorithm moves to $\mathbf{x} = \mathbf{e}_j$ where $j$ maximises $|z_j|$, and
the stopping test $|z_j| \le \mathbf{z}^T\mathbf{x}$ fires exactly when no vertex improves on
the current value.

Since $\mathbf{z}$ is the subgradient of $f$ at the current point, $|z_j| > \mathbf{z}^T\mathbf{x}$
guarantees $f(\mathbf{e}_j) > f(\mathbf{x})$. So the sequence of values is **strictly
increasing**, and since there are finitely many vertices it must terminate. $\square$

It terminates at a **local** maximum over vertices, which is why it can underestimate.

### 2.3 One step reduces the error by about $\kappa u$

Let $\mathbf{e} = \hat{\mathbf{x}} - \mathbf{x}$ be the error. With an **exact** residual,
$\mathbf{r} = -A\mathbf{e}$ exactly, so the correction solves $A\mathbf{d} = -A\mathbf{e}$,
whose exact answer is $\mathbf{d} = -\mathbf{e}$.

The solve is not exact: it is done with the existing factorization, so it returns
$\mathbf{d} + \delta\mathbf{d}$ with $\|\delta\mathbf{d}\|/\|\mathbf{d}\| = O(\kappa u)$. The
new error is $\mathbf{e} + \mathbf{d} + \delta\mathbf{d} = \delta\mathbf{d}$, so

$$\|\mathbf{e}_{\text{new}}\| \approx \kappa u\,\|\mathbf{e}_{\text{old}}\|. \qquad\square$$

**Convergence needs $\kappa u < 1$**, and the rate is $\kappa u$ per step.

**Measured**: at $\kappa = 10^{10}$, one step took the error from $2.39\times10^{-7}$ to
$5.61\times10^{-15}$, a factor of $2.3\times10^{-8}$ against a predicted $\kappa u \approx
1.1\times10^{-6}$. Better than predicted, because the bound is a worst case.

### 2.4 Refinement is Newton, and why it is only linear

Newton on $F(\mathbf{x}) = A\mathbf{x} - \mathbf{b}$ has $J = A$ and step
$-A^{-1}F(\mathbf{x}) = A^{-1}\mathbf{r}$, which is exactly the correction.

**Why not quadratic.** Newton's quadratic rate comes from the nonlinear remainder in Taylor's
theorem, which for a **linear** $F$ is exactly zero. In exact arithmetic refinement would
converge in **one** step. What limits it is not the mathematics but the rounding in the solve,
and that contributes a fixed relative factor $\kappa u$ each time, which is a linear rate.

So the rate is set by the arithmetic, not by the function.

### 2.5 How much extra precision the residual needs

To reach full working precision the residual must resolve quantities of relative size $\kappa u$
against operands of size 1, so it needs about

$$\log_{10}\kappa \text{ extra decimal digits}.$$

For $\kappa = 10^{12}$ that is 12 extra digits, so double precision residuals need roughly
double-double arithmetic. This is why `nalib` uses exact rational arithmetic in the lesson: it
supplies unlimited extra digits, so the limit shown is the technique's own rather than a
particular precision's.

### 3.1 The Higham-Tisseur estimator

Uses several starting vectors at once (a block method) and includes a vector designed to expose
patterns Hager's single-vector version misses.

It underestimates **less often** and by less, at a few times the cost, and is what LAPACK's
`lacn2` actually implements.

### 3.2 Double-double residuals

Use error-free transformations: `two_sum(a,b)` returns the sum and its exact rounding error, and
`two_product(a,b)` does the same for a product using FMA. Accumulating both parts gives about 32
digits.

It matches the rational result to well beyond what is needed, at a constant factor of about 20
rather than the large factor rationals cost. This is what production implementations use.

### 3.3 Mixed precision refinement

Factorize in `float32` ($O(n^3)$ at double the speed and half the memory), refine in `float64`
($O(n^2)$).

Modern GPUs make single precision 2 to 30 times faster than double, so the speedup is
substantial. The catch is the convergence condition, which becomes $\kappa u_{\text{single}}
< 1$, so $\kappa \lesssim 10^7$. Within that range you get double precision accuracy at single
precision cost, which is why every major GPU linear algebra library offers it.

### 4.1 Steps needed against $\kappa$

The count grows as $\kappa u$ approaches 1, because the per-step factor is $\kappa u$ and you
need $(\kappa u)^k$ to fall below $u$:

$$k \approx \frac{\log u}{\log(\kappa u)}.$$

For $\kappa u \ll 1$ that is 1 or 2 steps. As $\kappa u \to 1$ the denominator goes to zero and
the count diverges, which is the boundary section 5 measured.

### 4.2 When Hager does badly

It underestimates by more than 10 times only rarely for random matrices. It does worse on
matrices deliberately constructed to defeat it, which exist because the method is a local
search: Higham gives explicit families.

For practical matrices a factor of 10 is uncommon and a factor of 2 is typical, which section 2
measured (worst ratio 0.70).

### 4.3 Ill conditioning against pivot growth

Refinement repairs **growth** more completely.

- **Pivot growth** is a defect in the *factorization*, so the residual (computed from the
  original $A$) sees the error clearly and the correction removes it. Section 5 measured this on
  Wilkinson's matrix.
- **Ill conditioning** is a property of the *problem*. Refinement still converges to the exact
  solution of the stored system, but that system's answer is genuinely sensitive, so the
  achievable accuracy is limited by the data rather than the algorithm.

### 5.1 Skeel's analysis

> **Skeel's theorem.** One step of iterative refinement with a **working precision** residual
> makes the computed solution componentwise backward stable, provided the matrix is not too
> badly scaled.

So working-precision refinement is not useless: it improves the **componentwise backward
error**, even though it does not improve the forward error.

**No contradiction with section 4.** That experiment measured the *forward* error, which is
bounded by $\kappa u$ regardless of how good the backward error becomes. Skeel's result is about
a different quantity, and it explains why LAPACK offers working-precision refinement at all.

### 5.2 Refinement on tensor cores

`float16` has $u \approx 5\times10^{-4}$, so factorizing in half precision converges only for
$\kappa \lesssim 10^3$, which is too restrictive.

The practical scheme uses **three** precisions: `float16` for the factorization, `float32` for
the working solve, `float64` for the residual, with GMRES rather than a plain correction as the
inner solver to widen the convergence range. This reaches $\kappa \approx 10^8$ or so, and is
what Haidar and Tomov's work on GPU solvers implements.

### 5.3 Componentwise condition estimation

Estimate $\||A^{-1}||A||\mathbf{x}|\|_\infty$ by noting that
$|A^{-1}|\mathbf{y}$ for $\mathbf{y} \ge 0$ can be evaluated by solving with $A$ and taking
absolute values in the right order, so Hager's algorithm applies with minor changes.

For badly scaled matrices this gives a bound orders of magnitude sharper than the normwise one,
because it is invariant under row scaling (lesson 19 exercise 5.1). It is what LAPACK's
`gerfs` uses to report componentwise error bounds alongside a solution.

---

## Where these solutions sit in the course

Part 3 built the machinery the rest of the course assumes:

- **Norms and $\kappa$**, used for every error bound from here on.
- **Orthogonality**, which is why Parts 5 and 6 look the way they do: every stable algorithm
  there is built from reflections and rotations.
- **Factorization as an object**, not a procedure: factor once, solve many.
- **The separation of problem from algorithm**, made quantitative by the residual and $\kappa$.
- **Structure**, which is what makes large problems possible at all, and which Part 4 pushes to
  its conclusion by using only the matrix-vector product.
