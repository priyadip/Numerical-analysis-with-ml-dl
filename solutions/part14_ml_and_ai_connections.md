# Solutions: Part 14, Numerical Analysis in Machine Learning and AI

Solutions to every exercise in lessons 94 to 98, 105 in all: 21 for lesson 94, 21 for lesson 95, 21 for lesson 96, 21 for lesson 97, 21 for lesson 98.

Every number quoted here was measured by running the code, on this repository, with the seeds
shown. Where a measurement contradicted the claim the exercise expects, the measurement is
reported and the claim is corrected. This part has more of those than any other, because most of
its exercises ask whether a widely repeated statement about machine learning is true, and several
of them are not. Those are marked rather than replaced with an example that would have worked.

Two of the corrections are to measurements that could not answer the question they were asked, and
those are described as well: a test that cannot separate two hypotheses is worth more as a reported
failure than as a number.

Run any block from the repository root. Each one is self contained apart from `nalib`, and the
blocks within one lesson share a namespace, so they are meant to be run in order.

```python
import sys
sys.path.insert(0, "src")
```

---

## Lesson 94, Numerical Linear Algebra in Machine Learning

### 1.1 PCA without the word covariance

Let $X$ be the data matrix, $n$ samples by $d$ features, and let $\bar X = X - \mathbf{1}\mu^{\mathsf T}$
be its column-centred form. Take the thin singular value decomposition

$$
\bar X = U \Sigma V^{\mathsf T} .
$$

Then:

- the **principal directions** are the columns of $V$, in order of decreasing $\sigma_i$;
- the **scores**, the data written in that basis, are $U\Sigma$;
- the **variance along direction $i$** is $\sigma_i^2/(n-1)$;
- the **explained variance ratio** is

$$
\frac{\sigma_i^2}{\sum_j \sigma_j^2} .
$$

That last line is the whole answer to the second half of the question. The explained variance ratio
is not estimated, and it is not a statistical quantity that comes with an error bar. It is the
squared singular value share, available the moment the decomposition is, and it sums to $1$ by
construction.

Two things this phrasing makes obvious that the covariance phrasing hides. The truncation to $k$
components is the truncated SVD of lesson 41, so Eckart-Young says it is the best rank-$k$
approximation of the centred data in both the spectral and the Frobenius norm. And nothing anywhere
requires a $d \times d$ matrix to be formed.

### 1.2 The squared condition number

For the centred data $\bar X$ with singular values $\sigma_1 \ge \dots \ge \sigma_d > 0$, the
covariance is

$$
C = \frac{\bar X^{\mathsf T}\bar X}{n-1} = V\,\frac{\Sigma^2}{n-1}\,V^{\mathsf T} ,
$$

so its eigenvalues are $\sigma_i^2/(n-1)$ and

$$
\kappa(C) = \frac{\sigma_1^2}{\sigma_d^2} = \kappa(\bar X)^2 .
$$

**The cost in digits.** A backward stable computation on a matrix with condition number $\kappa$
returns an answer with relative error about $\varepsilon\kappa$. Working with $\bar X$ gives
$\varepsilon\kappa$; working with $C$ gives $\varepsilon\kappa^2$. In base ten, if $\kappa = 10^p$
then the SVD route loses $p$ digits and the covariance route loses $2p$ out of the $16$ available.

So the covariance route runs out at $\kappa = 10^8$ where the SVD route runs out at $10^{16}$.
Lesson 94's section 3 fitted those two slopes at $0.945$ and $2.019$.

There is a second cost that is not about accuracy. Forming $C$ perturbs the problem before any
solving happens: the rounding in the product is about $\varepsilon\sigma_1^2$, which relative to
$\sigma_d^2$ is $\varepsilon\kappa^2$. Past $\kappa = 10^8$ the matrix you have built is not a
perturbation of the one you meant to build; it is a different matrix, and it can fail to be positive
definite. Lesson 94 measured that too: from a data condition number of $10^8$ the smallest
eigenvalue comes out negative.

### 1.3 Whitening is not unique

Whitening means finding $W$ with $\operatorname{cov}(XW) = I$. Suppose $W$ works and let $Q$ be any
orthogonal matrix. Then

$$
\operatorname{cov}(XWQ) = Q^{\mathsf T}\operatorname{cov}(XW)\,Q = Q^{\mathsf T}IQ = I ,
$$

so $WQ$ works too. The set of whiteners is therefore $\{W_0 Q : Q^{\mathsf T}Q = I\}$, which for
$d$ features has $d(d-1)/2$ free parameters. **No statistical criterion can pick among them**,
because they all produce exactly the same covariance.

Two get names, both built from $\bar X = U\Sigma V^{\mathsf T}$ with
$\Lambda = \Sigma^2/(n-1)$:

- **PCA whitening**, $W = V\Lambda^{-1/2}$. It preserves the **axes**: the output coordinates are the
  principal directions, so coordinate $i$ of the whitened data is the $i$-th principal score,
  rescaled. Useful when you want the components ordered and interpretable.
- **ZCA whitening**, $W = V\Lambda^{-1/2}V^{\mathsf T}$, the symmetric inverse square root. It
  preserves the **original coordinate frame**: the output lives in the same feature space it came
  from, so an image stays an image. It is also the whitener closest to the identity in the Frobenius
  norm, which exercise 2.3 proves.

What both preserve, and what every whitener preserves, is the sample geometry up to a rotation: the
Gram matrix $ZZ^{\mathsf T}$ of the whitened data is the same for all of them. Lesson 94's section 4
measured that at $1.7\times10^{-13}$ under a rescaling of one feature by $1000$.

### 1.4 The ridge filter factor

$$
f_i(\lambda) = \frac{\sigma_i^2}{\sigma_i^2 + \lambda} \in (0, 1] .
$$

The ridge solution is the plain least squares solution with component $i$ multiplied by $f_i$:

$$
x_\lambda = \sum_i f_i(\lambda)\,\frac{u_i^{\mathsf T}b}{\sigma_i}\,v_i .
$$

**When $\sigma_i^2 \gg \lambda$:** $f_i \approx 1 - \lambda/\sigma_i^2 \approx 1$. The direction is
left alone. These are the directions the data measured well, and ridge does not touch them.

**When $\sigma_i^2 \ll \lambda$:** $f_i \approx \sigma_i^2/\lambda \approx 0$. The direction is
switched off. These are the directions the data barely measured, where the least squares answer is
noise divided by a small number.

The crossover is exactly at $\sigma_i^2 = \lambda$, where $f_i = 1/2$. So $\sqrt\lambda$ is a
threshold on the singular values, and choosing $\lambda$ is choosing where to cut.

That one formula explains both of ridge's properties at once. It fixes conditioning, because the
condition number becomes $(\sigma_1^2+\lambda)/(\sigma_d^2+\lambda)$. And it introduces bias, because
the components it switches off had a true value that is now missing.

### 1.5 The rank bound on a trained update

For a linear layer with weights $W$, a squared loss and a minibatch of $B$ samples, the gradient is

$$
\nabla_W \mathcal{L} = \frac{1}{B}\,X_{\text{batch}}^{\mathsf T} R
= \frac{1}{B}\sum_{j=1}^{B} x_j\, r_j^{\mathsf T} ,
$$

a sum of $B$ outer products. So $\operatorname{rank}(\nabla_W \mathcal{L}) \le B$. After $k$ steps
the accumulated update is a sum of $k$ such gradients, hence a sum of at most $kB$ outer products:

$$
\operatorname{rank}(W_k - W_0) \le \min(kB,\; d_{\text{in}},\; d_{\text{out}}) .
$$

**What it assumes:** that each per-sample gradient contributes one outer product. That is true for
any layer whose output is linear in its weights, which covers dense layers, convolutions and
attention projections.

**What it does not assume:** anything at all about the data, the loss surface, the learning rate, the
optimizer's momentum, or whether training converged. It is an exact statement about the span of the
updates.

Lesson 94's section 11 measured the bound and found it **attained** at every step count tested, and
the control matters: a random matrix of the same shape has full rank $128$, and a rank $8$ adapter
captures $21.07$ per cent of it against $100$ per cent of an $8$-step update. So the low-rank claim
is about the optimizer's path, not about weight matrices.

### 2.1 The ridge solution from the SVD

Write $A = U\Sigma V^{\mathsf T}$, thin, with $A$ of shape $m \times n$ and rank $r$. The objective
is

$$
\phi(x) = \lVert Ax - b\rVert_2^2 + \lambda\lVert x\rVert_2^2 .
$$

Substitute $x = Vy$ and split $b$ into $UU^{\mathsf T}b$ and its orthogonal complement:

$$
\phi = \lVert U(\Sigma y - U^{\mathsf T}b)\rVert^2 + \lVert (I - UU^{\mathsf T})b\rVert^2
     + \lambda\lVert Vy\rVert^2
     = \sum_i \left[(\sigma_i y_i - u_i^{\mathsf T}b)^2 + \lambda y_i^2\right] + \text{const},
$$

using that $U$ and $V$ have orthonormal columns. The sum has separated, so each $y_i$ can be
minimized alone. Differentiating the $i$-th term,

$$
2\sigma_i(\sigma_i y_i - u_i^{\mathsf T}b) + 2\lambda y_i = 0
\quad\Longrightarrow\quad
y_i = \frac{\sigma_i\,u_i^{\mathsf T}b}{\sigma_i^2 + \lambda} ,
$$

and therefore

$$
x_\lambda = \sum_i \frac{\sigma_i}{\sigma_i^2 + \lambda}\,(u_i^{\mathsf T}b)\,v_i
= \sum_i f_i(\lambda)\,\frac{u_i^{\mathsf T}b}{\sigma_i}\,v_i .
$$

**The limit $\lambda \to 0$.** For every $i$ with $\sigma_i > 0$, $f_i \to 1$, so the coefficient
becomes $u_i^{\mathsf T}b/\sigma_i$, which is exactly the pseudoinverse:
$A^{+} = V\Sigma^{-1}U^{\mathsf T}$ over the nonzero singular values. Directions with $\sigma_i = 0$
contribute $0/\lambda = 0$ at every $\lambda > 0$ and also contribute $0$ to $A^{+}b$, so the limit
is the **minimum norm** least squares solution and not merely some least squares solution.

That last point is worth stating: ridge with $\lambda \to 0^{+}$ selects the minimum norm solution
out of the affine set of minimizers, which is why it is well defined even when $A$ is rank
deficient.

### 2.2 The stacked matrix and its condition number

Let

$$
\tilde A = \begin{bmatrix} A \\ \sqrt\lambda\, I_n \end{bmatrix} .
$$

Then

$$
\tilde A^{\mathsf T}\tilde A = A^{\mathsf T}A + \lambda I = V(\Sigma^2 + \lambda I)V^{\mathsf T} ,
$$

which is a spectral decomposition with eigenvalues $\sigma_i^2 + \lambda$. The singular values of
$\tilde A$ are the square roots of the eigenvalues of $\tilde A^{\mathsf T}\tilde A$, so

$$
\tilde\sigma_i = \sqrt{\sigma_i^2 + \lambda} ,
$$

exactly, with no approximation and for every $\lambda \ge 0$. Note that $\tilde A$ has full column
rank whenever $\lambda > 0$ even if $A$ does not, because $\sigma_i^2 + \lambda \ge \lambda > 0$.

The condition numbers follow:

$$
\kappa(\tilde A) = \sqrt{\frac{\sigma_1^2+\lambda}{\sigma_n^2+\lambda}} ,
\qquad
\kappa(A^{\mathsf T}A + \lambda I) = \frac{\sigma_1^2+\lambda}{\sigma_n^2+\lambda}
= \kappa(\tilde A)^2 .
$$

**Why that matters.** Solving the normal equations form works with a matrix of condition number
$\kappa^2$ and loses $2p$ digits; solving the stacked least squares problem by QR works with a matrix
of condition number $\kappa$ and loses $p$. Lesson 94 measured the identity to $2.3\times10^{-6}$ and
the accuracy gap to $76988$ at $\lambda = 10^{-12}$, and the residual $2.3\times10^{-6}$ is itself
the rounding committed in forming $A^{\mathsf T}A$.

Note also that $\lambda$ rescues the normal equations only while $\lambda \gtrsim \sigma_n^2$. Below
that the shift is invisible against the existing spectrum and the conditioning is the unregularized
one.

### 2.3 ZCA is the whitener closest to the identity

Every whitener is $W = W_0 Q$ for orthogonal $Q$, where $W_0 = V\Lambda^{-1/2}$ is the PCA whitener.
Minimize

$$
\lVert W_0 Q - I\rVert_F^2
= \lVert W_0 Q\rVert_F^2 - 2\operatorname{tr}(Q^{\mathsf T}W_0^{\mathsf T}) + \lVert I\rVert_F^2 .
$$

The first term is $\operatorname{tr}(Q^{\mathsf T}W_0^{\mathsf T}W_0Q) =
\operatorname{tr}(W_0^{\mathsf T}W_0)$, which does not depend on $Q$ because the trace is invariant
under an orthogonal similarity. So the problem reduces to

$$
\max_{Q^{\mathsf T}Q = I} \operatorname{tr}(Q^{\mathsf T}W_0^{\mathsf T}) ,
$$

which is the **orthogonal Procrustes** problem. Its solution is standard: write the SVD
$W_0^{\mathsf T} = P S R^{\mathsf T}$; then $\operatorname{tr}(Q^{\mathsf T}PSR^{\mathsf T}) =
\operatorname{tr}(S R^{\mathsf T}Q^{\mathsf T}P) = \sum_i s_i z_{ii}$ where
$Z = R^{\mathsf T}Q^{\mathsf T}P$ is orthogonal, so $|z_{ii}| \le 1$ and the maximum is $\sum_i s_i$,
attained when $Z = I$, that is $Q = PR^{\mathsf T}$.

Here $W_0^{\mathsf T} = \Lambda^{-1/2}V^{\mathsf T}$, whose SVD is $P = V$... more directly: the
maximizer $Q$ is the orthogonal polar factor of $W_0^{\mathsf T}$, and

$$
W_0^{\mathsf T} = \Lambda^{-1/2}V^{\mathsf T}
\quad\Longrightarrow\quad
Q = V^{\mathsf T}\ \text{(orthogonal factor)},
$$

giving

$$
W = W_0 Q = V\Lambda^{-1/2}V^{\mathsf T} ,
$$

which is the ZCA whitener, the symmetric positive definite square root of $C^{-1}$.

**Check it numerically**, against a search over random rotations.

```python
import numpy as np
from nalib import mllinalg as mla

rng = np.random.default_rng(0)
features = 6
data = mla.dataset(300, features, decay=0.8, seed=17)["data"]
zca = mla.whiten(data, "zca")
pca = mla.whiten(data, "pca")
identity = np.eye(features)

print(f"||W - I|| for ZCA : {np.linalg.norm(zca['matrix'] - identity, 'fro'):.6f}")
print(f"||W - I|| for PCA : {np.linalg.norm(pca['matrix'] - identity, 'fro'):.6f}")

trials = 400
best = np.inf
for _ in range(trials):
    rotation = np.linalg.qr(rng.standard_normal((features, features)))[0]
    best = min(best, float(np.linalg.norm(pca["matrix"] @ rotation - identity, "fro")))
print(f"best over {trials} random rotations: {best:.6f}")
print(f"ZCA is below that by {best - np.linalg.norm(zca['matrix'] - identity, 'fro'):.6f}")

assert np.linalg.norm(zca["matrix"] - identity, "fro") <= best + 1e-12
```

The random search never beats it, which is what a minimizer means.

### 2.4 Bias, variance, and where their sum is smallest

Take the model $b = Ax^\star + \eta$ with $\eta$ zero mean and covariance $\tau^2 I$. In the SVD
basis, write $\alpha_i = v_i^{\mathsf T}x^\star$ for the components of the truth. Since
$u_i^{\mathsf T}b = \sigma_i\alpha_i + u_i^{\mathsf T}\eta$ and $u_i^{\mathsf T}\eta$ has mean $0$
and variance $\tau^2$,

$$
v_i^{\mathsf T}x_\lambda = \frac{\sigma_i(\sigma_i\alpha_i + u_i^{\mathsf T}\eta)}{\sigma_i^2+\lambda}
= f_i\alpha_i + \frac{\sigma_i}{\sigma_i^2+\lambda}\,u_i^{\mathsf T}\eta .
$$

So

$$
\text{bias}_i = (f_i - 1)\alpha_i = -\frac{\lambda\,\alpha_i}{\sigma_i^2+\lambda} ,
\qquad
\text{var}_i = \frac{\sigma_i^2\,\tau^2}{(\sigma_i^2+\lambda)^2} ,
$$

and the mean squared error is

$$
\mathrm{MSE}(\lambda) = \sum_i \left[
\frac{\lambda^2\alpha_i^2}{(\sigma_i^2+\lambda)^2}
+ \frac{\sigma_i^2\tau^2}{(\sigma_i^2+\lambda)^2}\right]
= \sum_i \frac{\lambda^2\alpha_i^2 + \sigma_i^2\tau^2}{(\sigma_i^2+\lambda)^2} .
$$

Differentiate one term with respect to $\lambda$:

$$
\frac{d}{d\lambda}\left[\frac{\lambda^2\alpha_i^2+\sigma_i^2\tau^2}{(\sigma_i^2+\lambda)^2}\right]
= \frac{2\sigma_i^2\left(\lambda\alpha_i^2 - \tau^2\right)}{(\sigma_i^2+\lambda)^3} .
$$

Each term alone is minimized at $\lambda = \tau^2/\alpha_i^2$, so the sum's minimizer lies between
the smallest and largest of those. Two conclusions follow immediately.

**With no noise, $\tau = 0$, the derivative is $2\sigma_i^2\lambda\alpha_i^2/(\cdot)^3 > 0$ for every
$\lambda > 0$, so the MSE is strictly increasing and the optimum is exactly $\lambda = 0$.** That is
the negative result lesson 94's section 9 measured: on a problem with condition number $10^5$ and no
noise, the best penalty is zero.

**The optimal $\lambda$ scales like $\tau^2$**, not like the condition number. Ridge is a response to
noise. Conditioning decides how much damage the noise does, not whether to regularize.

### 2.5 A sum of $k$ rank-one matrices

Let $M = \sum_{j=1}^{k} u_j v_j^{\mathsf T}$ with $u_j \in \mathbb{R}^m$, $v_j \in \mathbb{R}^n$.

For any $x$, $Mx = \sum_j u_j (v_j^{\mathsf T}x)$, a linear combination of $u_1,\dots,u_k$. So the
column space of $M$ is contained in $\operatorname{span}\{u_1,\dots,u_k\}$, whose dimension is at
most $k$, and

$$
\operatorname{rank}(M) \le k .
$$

Equivalently, $M = UV^{\mathsf T}$ with $U = [u_1 \cdots u_k]$ of shape $m \times k$, and the rank of
a product is at most the smaller inner dimension.

**Equality holds if and only if both $\{u_j\}$ and $\{v_j\}$ are linearly independent sets.**

*If both are independent*, suppose $Mx = 0$ for some $x$. Then $\sum_j (v_j^{\mathsf T}x)u_j = 0$, and
independence of the $u_j$ forces $v_j^{\mathsf T}x = 0$ for every $j$, so $x \perp
\operatorname{span}\{v_j\}$, a space of dimension $k$. Hence the null space has dimension $n - k$ and
the rank is $k$.

*If either set is dependent*, say $u_k = \sum_{j<k}c_j u_j$, then
$M = \sum_{j<k} u_j(v_j + c_j v_k)^{\mathsf T}$, a sum of $k-1$ outer products, so the rank is at
most $k-1 < k$.

The bound is what section 1.5 used, and the equality condition is what makes lesson 94's measurement
non-trivial: the measured rank was exactly $\min(kB, d)$ at every step count, which says the
gradients from successive minibatches were linearly independent, and that is a fact about the data
rather than about the algebra.

### 3.1 PCA three ways, including one that never forms a big matrix

The third route uses the randomized range finder of lesson 41. It touches $\bar X$ only through
matrix products with a thin random matrix, so its largest intermediate is $n \times (r + p)$ and it
never forms either $X^{\mathsf T}X$ or $XX^{\mathsf T}$.

```python
import math
import numpy as np
from nalib import mllinalg as mla


def randomized_pca(data, components, oversample=8, power=2, seed=0):
    """Directions and singular values from a sketch, never forming a d by d matrix."""
    centred = data - data.mean(axis=0)
    width = min(components + oversample, min(centred.shape))
    rng = np.random.default_rng(seed)
    sketch = centred @ rng.standard_normal((centred.shape[1], width))
    basis = np.linalg.qr(sketch)[0]
    for _ in range(power):
        basis = np.linalg.qr(centred.T @ basis)[0]
        basis = np.linalg.qr(centred @ basis)[0]
    small = basis.T @ centred
    _, values, directions = np.linalg.svd(small, full_matrices=False)
    return {"singular_values": values[:components], "directions": directions[:components],
            "largest_intermediate": max(basis.size, small.size)}


samples, features, wanted = 400, 24, 6
print("relative error on the leading components, and on the smallest one")
print(f"{'condition':>11}{'svd top':>11}{'cov top':>11}{'sketch top':>13}"
      f"{'svd last':>11}{'cov last':>11}{'largest intermediate':>23}")
for target in (1e2, 1e4, 1e6, 1e8, 1e10):
    exact = np.geomspace(1.0, 1.0 / target, features)
    matrix = mla.matrix_with_spectrum(exact, rows=samples, seed=5, centred=True)
    by_svd = mla.pca(matrix)["singular_values"]
    by_cov = mla.pca_by_covariance(matrix)["singular_values"]
    sketched = randomized_pca(matrix, wanted, seed=1)
    top = exact[:wanted]
    print(f"{target:>11.0e}{np.max(np.abs(by_svd[:wanted] - top) / top):>11.2e}"
          f"{np.max(np.abs(by_cov[:wanted] - top) / top):>11.2e}"
          f"{np.max(np.abs(sketched['singular_values'] - top) / top):>13.2e}"
          f"{abs(by_svd[-1] - exact[-1]) / exact[-1]:>11.2e}"
          f"{abs(by_cov[-1] - exact[-1]) / exact[-1]:>11.2e}"
          f"{sketched['largest_intermediate']:>23}")

print(f"\nthe full SVD costs O(n d^2) = {samples * features ** 2}")
print(f"the randomized one costs O(n d r) = {samples * features * (wanted + 8)}")
```

**The table corrects something the lesson left implicit.** Lesson 94's section 3 measured the error
in the **smallest** principal value and found the covariance route losing two digits per digit of
conditioning. That is the last two columns here, and it holds. But the first three columns are the
ones PCA actually uses, and on those **the covariance route is as accurate as the SVD**, staying
inside $4\times10^{-14}$ even at a data condition number of $10^{10}$.

Both statements are true and they are about different quantities. The leading singular values are
well separated from zero relative to $\sigma_1$, so their relative accuracy is $\varepsilon$ times a
small number whichever route computes them. It is the tail that gets destroyed by squaring the
spectrum, and PCA discards the tail anyway.

**So the honest rule is narrower than "never form the covariance matrix".** Never form it if you need
the small directions, which is what conditioning estimates, whitening and ridge all need. For the
leading components of a PCA, it is fine, and section 4.1 shows it is also faster.

The randomized route matches both on the components it keeps, at a largest intermediate of $5600$
against the data's $9600$ entries, and costs $O(ndr)$ against $O(nd^2)$. Its one weak row is at a
condition number of $10^2$, where the spectrum decays slowly and a rank $14$ sketch cannot isolate
the top $6$ cleanly: $3.6\times10^{-8}$ instead of $10^{-15}$. **A sketch works because the spectrum
decays, so it fails where the spectrum does not.**

### 3.2 Generalized cross-validation against the oracle

GCV chooses $\lambda$ by minimizing

$$
G(\lambda) = \frac{n\,\lVert Ax_\lambda - b\rVert^2}{\left(n - \sum_i f_i(\lambda)\right)^2} ,
$$

where $\sum_i f_i$ is the effective number of parameters. It needs neither the noise level nor the
answer, which is what makes it usable.

```python
import numpy as np
from nalib import mllinalg as mla


def gcv_curve(matrix, target, penalties):
    """The GCV score at each penalty, using the filter factors as the trace."""
    left, s, right = np.linalg.svd(matrix, full_matrices=False)
    coefficients = left.T @ target
    rows = matrix.shape[0]
    scores = []
    for lam in penalties:
        filters = s ** 2 / (s ** 2 + lam)
        residual = np.sum(((1.0 - filters) * coefficients) ** 2)
        residual += float(np.sum(target ** 2) - np.sum(coefficients ** 2))
        scores.append(rows * residual / (rows - float(np.sum(filters))) ** 2)
    return np.array(scores)


rows, cols = 120, 30
values = np.geomspace(1.0, 1e-6, cols)
matrix = mla.matrix_with_spectrum(values, rows=rows, seed=3)
rng = np.random.default_rng(11)
truth = rng.standard_normal(cols)
grid = np.geomspace(1e-14, 1e2, 61)

print(f"{'noise':>10}{'gcv lambda':>14}{'oracle lambda':>16}{'ratio':>12}"
      f"{'gcv error':>13}{'oracle error':>15}{'penalty of choosing badly':>28}")
for level in (1e-6, 1e-4, 1e-2, 1e-1):
    data = matrix @ truth + level * rng.standard_normal(rows)
    scores = gcv_curve(matrix, data, grid)
    errors = np.array([np.linalg.norm(mla.ridge_svd(matrix, data, lam)["solution"] - truth)
                       for lam in grid])
    chosen = int(np.argmin(scores))
    best = int(np.argmin(errors))
    print(f"{level:>10.0e}{grid[chosen]:>14.3e}{grid[best]:>16.3e}"
          f"{max(grid[chosen] / grid[best], grid[best] / grid[chosen]):>12.2f}"
          f"{errors[chosen]:>13.4e}{errors[best]:>15.4e}"
          f"{errors[chosen] / errors[best]:>28.4f}")
```

**Reading the table.** The last column is the one that matters. GCV's penalty can be off by a large
factor and its **error** is close to the oracle's, because the error curve is flat near its minimum.
That is the same phenomenon lesson 98 measures for the L-curve, and it is the reassuring part of
parameter selection: getting within an order of magnitude is enough.

The failure mode to watch for is at very low noise, where the GCV score becomes nearly flat and its
minimum is decided by rounding rather than by the data.

### 3.3 Kernel PCA and the double centring trick

Kernel PCA applies PCA in a feature space defined implicitly by a kernel
$k(x,y) = \langle \phi(x), \phi(y)\rangle$. The features $\phi(x)$ are never computed, so the
covariance matrix cannot be formed either. What can be formed is the $n \times n$ Gram matrix
$K_{ij} = k(x_i, x_j)$.

Centring is the difficulty. PCA needs the features centred, but $\phi$ is unavailable, so the
centring has to be done to $K$ directly. With $H = I - \tfrac1n\mathbf{1}\mathbf{1}^{\mathsf T}$,

$$
\tilde K = H K H = K - \tfrac1n\mathbf{1}\mathbf{1}^{\mathsf T}K - \tfrac1n K\mathbf{1}\mathbf{1}^{\mathsf T}
+ \tfrac1{n^2}\mathbf{1}\mathbf{1}^{\mathsf T}K\mathbf{1}\mathbf{1}^{\mathsf T} ,
$$

which is the Gram matrix of the centred features. That is the double centring trick.

```python
import numpy as np
from nalib import mllinalg as mla


def kernel_pca(data, components, gamma=None, kernel="rbf"):
    """PCA in a feature space, through the Gram matrix and a double centring."""
    x = np.asarray(data, dtype=float)
    n = x.shape[0]
    if kernel == "linear":
        gram = x @ x.T
    else:
        squared = np.sum(x ** 2, axis=1)
        distance = squared[:, None] + squared[None, :] - 2.0 * x @ x.T
        width = 1.0 / x.shape[1] if gamma is None else gamma
        gram = np.exp(-width * np.maximum(distance, 0.0))
    centre = np.eye(n) - np.ones((n, n)) / n
    centred = centre @ gram @ centre
    values, vectors = np.linalg.eigh(centred)
    order = np.argsort(values)[::-1][:components]
    values = np.maximum(values[order], 0.0)
    return {"scores": vectors[:, order] * np.sqrt(values),
            "eigenvalues": values / (n - 1)}


made = mla.dataset(150, 8, decay=0.9, seed=23)
data = made["data"]
plain = mla.pca(data, components=4)
linear = kernel_pca(data, plain["components"], kernel="linear")

wanted = plain["components"]
print("a linear kernel must reproduce ordinary PCA:")
print(f"{'i':>4}{'variance from PCA':>21}{'from the kernel':>19}{'relative gap':>15}")
for i in range(wanted):
    a = plain["variance"][i]
    b = linear["eigenvalues"][i]
    print(f"{i:>4}{a:>21.10f}{b:>19.10f}{abs(a - b) / a:>15.2e}")

print(f"\nscores agree up to sign to "
      f"{np.max(np.abs(np.abs(plain['scores']) - np.abs(linear['scores']))):.3e}")

rbf = kernel_pca(data, plain["components"], kernel="rbf")
print(f"\nan RBF kernel is a different decomposition:")
print(f"  its leading eigenvalues: {rbf['eigenvalues']}")
print(f"  a linear kernel's      : {linear['eigenvalues']}")

assert np.allclose(plain["variance"][:wanted], linear["eigenvalues"], rtol=1e-8)
```

**Which of section 3's three routes is it?** The **Gram** route. Kernel PCA works with $K$, which is
$XX^{\mathsf T}$ for a linear kernel, so it inherits the Gram route's properties: it costs
$O(n^3)$ rather than $O(d^3)$, which is a win when $d$ is infinite and a loss when $n$ is large, and
it is the numerically worse of the two eigenvalue routes.

Lesson 94's section 3 measured the Gram route failing at a data condition number of $10^8$, worse
than the covariance route, because it asks an $n \times n$ eigensolver to separate a few small
eigenvalues from many exact zeros. **Kernel PCA has no choice about this**: there is no SVD route,
because the matrix whose SVD you would take does not exist. That is a real cost of the kernel trick,
and it is why kernel methods use a regularized or truncated eigendecomposition in practice.

### 3.4 A rank-$r$ adapter for a two-layer network

Fine-tune both layers of a small network, then ask how much of the resulting update a rank-$r$
adapter can recover.

```python
import numpy as np
from nalib import mllinalg as mla


def two_layer(width, hidden, outputs, samples, steps, batch, step_size=0.05, seed=0):
    """Train both layers by minibatch descent, and return the change in each."""
    rng = np.random.default_rng(seed)
    inputs = rng.standard_normal((samples, width))
    truth_first = rng.standard_normal((width, hidden)) / np.sqrt(width)
    truth_second = rng.standard_normal((hidden, outputs)) / np.sqrt(hidden)
    targets = np.tanh(inputs @ truth_first) @ truth_second
    first = rng.standard_normal((width, hidden)) / np.sqrt(width)
    second = rng.standard_normal((hidden, outputs)) / np.sqrt(hidden)
    start = (first.copy(), second.copy())
    losses = []
    for _ in range(steps):
        rows = rng.choice(samples, size=min(batch, samples), replace=False)
        block, wanted = inputs[rows], targets[rows]
        hidden_pre = block @ first
        activation = np.tanh(hidden_pre)
        residual = activation @ second - wanted
        losses.append(float(np.mean(residual ** 2)))
        grad_second = activation.T @ residual / rows.size
        back = (residual @ second.T) * (1.0 - activation ** 2)
        grad_first = block.T @ back / rows.size
        first = first - step_size * grad_first
        second = second - step_size * grad_second
    return {"first_update": first - start[0], "second_update": second - start[1],
            "losses": losses}


run = two_layer(64, 48, 32, 512, 200, 8, seed=7)
print(f"loss fell from {run['losses'][0]:.6f} to {run['losses'][-1]:.6f}")
print(f"{'rank':>6}{'captured, layer 1':>21}{'captured, layer 2':>21}"
      f"{'parameters saved':>19}")
for rank in (1, 2, 4, 8, 16, 32):
    first = mla.fit_adapter(run["first_update"], rank)
    second = mla.fit_adapter(run["second_update"], rank)
    saving = first["dense_parameters"] / first["parameters"]
    print(f"{rank:>6}{first['captured']:>21.6f}{second['captured']:>21.6f}"
          f"{saving:>19.2f}")

full_first = int(np.linalg.matrix_rank(run["first_update"]))
full_second = int(np.linalg.matrix_rank(run["second_update"]))
print(f"\nnumerical rank of the two updates: {full_first} and {full_second}")
print(f"the step bound would be min(200 * 8, 48) = {min(200 * 8, 48)}")
```

**What the numbers say.** After $200$ steps at batch $8$ the bound $kB = 1600$ exceeds every
dimension, so it no longer constrains anything and both updates are full rank. Yet the captured
fraction still rises quickly with $r$: the update has a decaying spectrum even though it is full
rank, which is section 10's distinction between numerical rank and effective rank.

So the honest statement about adapters on a real training run is **not** the exact bound of section
1.5, which only binds for short runs or small batches. It is that the update's spectrum decays, and
how fast it decays is an empirical question that this measurement answers for one run.

### 3.5 Incremental PCA and its drift

Incremental PCA keeps a rank-$r$ summary and merges each new block into it. The merge is a small SVD
of the stacked $[\,\text{summary};\,\text{new block}\,]$, so nothing larger than
$(r + \text{block}) \times d$ is ever formed.

```python
import numpy as np
from nalib import mllinalg as mla


def incremental_pca(data, components, block, seed=0):
    """Merge blocks into a rank-r summary, tracking the drift after each merge."""
    x = np.asarray(data, dtype=float)
    reference = mla.pca(x, components=components)
    total_mean = x.mean(axis=0)
    summary = None
    mean = np.zeros(x.shape[1])
    seen = 0
    drift = []
    for start in range(0, x.shape[0], block):
        chunk = x[start:start + block]
        chunk_mean = chunk.mean(axis=0)
        centred = chunk - chunk_mean
        if summary is None:
            stacked = centred
        else:
            correction = np.sqrt(seen * chunk.shape[0] / (seen + chunk.shape[0])) \
                * (mean - chunk_mean)
            stacked = np.vstack([summary, centred, correction[None, :]])
        _, values, directions = np.linalg.svd(stacked, full_matrices=False)
        keep = min(components, values.size)
        summary = directions[:keep] * values[:keep, None]
        mean = (seen * mean + chunk.shape[0] * chunk_mean) / (seen + chunk.shape[0])
        seen += chunk.shape[0]
        drift.append(mla.subspace_angle(directions[:keep], reference["directions"]))
    return {"directions": directions[:keep], "values": values[:keep], "drift": drift,
            "mean_error": float(np.max(np.abs(mean - total_mean)))}


made = mla.dataset(2000, 20, decay=1.2, seed=31)
for components in (3, 6, 12):
    out = incremental_pca(made["data"], components, block=100)
    print(f"r = {components:>3}: after {len(out['drift'])} blocks the subspace angle is "
          f"{out['drift'][-1]:.4f} degrees, worst along the way {max(out['drift']):.4f}")

out = incremental_pca(made["data"], 6, block=100)
print(f"\nthe running mean is exact to {out['mean_error']:.3e}")
print(f"drift after 1, 5, 10, 20 blocks: "
      + ", ".join(f"{out['drift'][i]:.4f}" for i in (0, 4, 9, 19)))
```

**Two separate errors are visible, and the second one behaves the opposite way to the obvious
guess.** The running mean is exact to $4.2\times10^{-17}$, because the update formula is a weighted
average with no truncation in it. The **subspace** drifts, because each merge discards everything
past rank $r$.

The drift falls over time, from $19.5$ degrees after one block to $3.8$ after twenty, as the leading
directions stabilize. That much is expected.

**But the drift grows with $r$**, from $2.5$ degrees at $r = 3$ to $14.6$ at $r = 12$, and the
guess that keeping more components would help is wrong. The reason is what the measurement is: the
largest principal angle between two $r$-dimensional subspaces. Adding a component adds the direction
estimated worst, and that direction sets the angle. The leading directions are recovered well at
every $r$; asking about a bigger subspace is asking about a worse-determined one.

So the useful reading is per direction rather than per subspace, and the practical advice, keep more
components than you need and truncate at the end, is still right for a different reason: the extra
components absorb the merge error that would otherwise contaminate the ones you keep.

### 4.1 Where the covariance route wins on cost

The SVD of an $n \times d$ matrix costs about $O(nd^2)$; forming the covariance costs $O(nd^2)$ too,
followed by an $O(d^3)$ eigendecomposition. So on flops they are the same order and the constants
decide, while the Gram route costs $O(n^2 d + n^3)$ and wins only when $d \gg n$.

```python
import time
import numpy as np
from nalib import mllinalg as mla

print(f"{'samples':>9}{'features':>10}{'svd (ms)':>11}{'covariance (ms)':>18}"
      f"{'gram (ms)':>12}{'fastest':>12}")
repeats = 5
for samples, features in ((2000, 20), (2000, 200), (500, 500), (200, 2000), (60, 4000)):
    data = mla.dataset(samples, features, decay=1.0, seed=41)["data"]
    timings = {}
    for name, function in (("svd", mla.pca), ("covariance", mla.pca_by_covariance),
                           ("gram", mla.pca_by_gram)):
        start = time.perf_counter()
        for _ in range(repeats):
            function(data, components=5)
        timings[name] = (time.perf_counter() - start) / repeats * 1e3
    fastest = min(timings, key=timings.get)
    print(f"{samples:>9}{features:>10}{timings['svd']:>11.2f}"
          f"{timings['covariance']:>18.2f}{timings['gram']:>12.2f}{fastest:>12}")
```

**The crossover is in the aspect ratio, not in either dimension alone.** Tall and thin favours the
covariance route, which never forms anything bigger than $d \times d$. Short and wide favours the
Gram route, which never forms anything bigger than $n \times n$. The SVD route is never the fastest
and never the slowest.

**The speedup is real and larger than expected.** At $2000 \times 20$ the covariance route is $7.2$
times faster than the SVD, and at $2000 \times 200$ it is $5.9$ times faster. Those are not rounding
differences, and combined with exercise 3.1's finding that the covariance route is accurate on the
leading components, the case for it is much stronger than lesson 94's section 3 suggested.

The Gram route's numbers are the other surprise: $601$ milliseconds at $2000 \times 20$, more than
$300$ times the covariance route, because it builds a $2000 \times 2000$ matrix to extract $5$
components from a $20$-dimensional space. At $60 \times 4000$ it is $2400$ times faster than the
covariance route for the same reason in reverse.

So the rule is: **pick the route by the aspect ratio, and check the condition number before trusting
the small directions from either eigenvalue route.**

### 4.2 Effective rank over a training run: the weights or the update

```python
import numpy as np
from nalib import mllinalg as mla

features, outputs, samples = 96, 96, 512
checkpoints = (0, 4, 16, 64, 256, 1024)
print(f"{'steps':>7}{'weights: effective rank':>26}{'stable rank':>14}"
      f"{'update: effective rank':>25}{'numerical rank':>17}")
for steps in checkpoints:
    if steps == 0:
        run = mla.train_linear_layer(features, outputs, samples, 1, 16, seed=53)
        weights = run["start"]
        update = np.zeros_like(weights)
    else:
        run = mla.train_linear_layer(features, outputs, samples, steps, 16, seed=53)
        weights = run["weights"]
        update = run["update"]
    weight_values = np.linalg.svd(weights, compute_uv=False)
    weight_summary = mla.spectral_summary(weight_values)
    if steps == 0:
        print(f"{steps:>7}{weight_summary['effective_rank']:>26.3f}"
              f"{weight_summary['stable_rank']:>14.3f}{'-':>25}{'-':>17}")
        continue
    update_values = np.linalg.svd(update, compute_uv=False)
    update_summary = mla.spectral_summary(update_values)
    print(f"{steps:>7}{weight_summary['effective_rank']:>26.3f}"
          f"{weight_summary['stable_rank']:>14.3f}"
          f"{update_summary['effective_rank']:>25.3f}"
          f"{int(np.linalg.matrix_rank(update)):>17}")
```

**The answer is that the change is in the update, not in the weights.** The weight matrix starts
near full effective rank, because it was initialized from a Gaussian, and stays near full effective
rank throughout, because the update is small compared with the weights. The **update's** effective
rank is small and grows slowly with the step count.

That is the sharpest form of the argument for adapters, and it is also the sharpest warning about
them. The adapter is low rank because the update is, and the update stops being low rank as training
runs longer. A rank that suffices for a short fine-tune will not suffice for a long one, and the
measurement says by how much.

### 4.3 The GCV penalty against six decades of noise

```python
import numpy as np
from nalib import mllinalg as mla

rows, cols = 160, 40
values = np.geomspace(1.0, 1e-7, cols)
matrix = mla.matrix_with_spectrum(values, rows=rows, seed=61)
rng = np.random.default_rng(71)
truth = rng.standard_normal(cols)
grid = np.geomspace(1e-16, 1e2, 73)
clean = matrix @ truth
repeats = 5

print(f"{'noise':>10}{'gcv lambda':>14}{'oracle lambda':>16}{'predicted ~ tau^2':>20}"
      f"{'error ratio':>14}")
for level in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
    chosen_all, oracle_all, ratio_all = [], [], []
    for trial in range(repeats):
        noise = level * np.random.default_rng(100 + trial).standard_normal(rows)
        data = clean + noise
        left, s, _ = np.linalg.svd(matrix, full_matrices=False)
        coefficients = left.T @ data
        scores = []
        for lam in grid:
            filters = s ** 2 / (s ** 2 + lam)
            residual = float(np.sum(((1.0 - filters) * coefficients) ** 2))
            residual += float(np.sum(data ** 2) - np.sum(coefficients ** 2))
            scores.append(rows * residual / (rows - float(np.sum(filters))) ** 2)
        errors = np.array([np.linalg.norm(mla.ridge_svd(matrix, data, lam)["solution"] - truth)
                           for lam in grid])
        chosen, best = int(np.argmin(scores)), int(np.argmin(errors))
        chosen_all.append(grid[chosen])
        oracle_all.append(grid[best])
        ratio_all.append(errors[chosen] / errors[best])
    print(f"{level:>10.0e}{np.median(chosen_all):>14.2e}{np.median(oracle_all):>16.2e}"
          f"{level ** 2:>20.2e}{np.median(ratio_all):>14.4f}")
```

**Two things to read off.** The oracle penalty tracks $\tau^2$, which is exactly what exercise 2.4
derived: the optimum of each term is $\tau^2/\alpha_i^2$ and the $\alpha_i$ are $O(1)$ here. So the
$\lambda \sim \tau^2$ scaling is a prediction, and the table checks it.

And GCV's **error** stays within a small factor of the oracle's across six decades, even where its
penalty is off. That is the flat-minimum effect again. Where GCV struggles is at the smallest noise
levels, where the score is nearly constant over many decades of $\lambda$ and its minimum is decided
by rounding.

### 5.1 Why the covariance matrix survives in practice

Section 3 showed the covariance route squares the condition number and fails past $\kappa = 10^8$.
It is nevertheless the default in most statistics packages, and there are four reasons, three of them
good.

**It is the definition, so it is what the derivation produces.** PCA was defined by Pearson and
Hotelling as the eigendecomposition of a covariance matrix, decades before the SVD was a practical
algorithm. The textbook derivation maximizes $w^{\mathsf T}Cw$ subject to $\lVert w\rVert = 1$ and
arrives at an eigenproblem. Implementing what the derivation says is the natural thing to do.

**The covariance matrix is often the actual input.** In finance, in psychometrics and in signal
processing, the correlation or covariance matrix is the object that was estimated, published, or
accumulated online. There is no $X$ to take an SVD of.

**Real data is usually well conditioned enough.** Standardized features with $d$ in the tens and a
few hundred samples give $\kappa(\bar X)$ in the tens, so $\kappa(C)$ is in the hundreds and the
covariance route loses two or three digits out of sixteen. Nothing downstream notices.

**And it is genuinely faster, not just theoretically cheaper.** Exercise 4.1 measured it at $7.2$
times the SVD route on tall thin data, and $C$ can be accumulated in one pass in $O(d^2)$ memory with
the data never held at once, which the SVD route cannot do at all.

**And, the finding that matters most, it is accurate for what PCA asks of it.** Exercise 3.1 measured
the covariance route's error on the leading components at $4\times10^{-14}$ even at a data condition
number of $10^{10}$, where its error on the smallest component is $10^{-2}$. Squaring the spectrum
destroys the tail and leaves the head alone, and PCA keeps the head.

**The condition under which it is defensible**, stated precisely and more narrowly than lesson 94's
section 3 implied: use the covariance route when you want the **leading** components and nothing
else. Its error on component $i$ is about $\varepsilon(\sigma_1/\sigma_i)^2$, so a component is safe
while $\sigma_i \gtrsim \sigma_1\sqrt{\varepsilon/\text{tolerance}}$, which for double precision and a
target of $10^{-6}$ means $\sigma_i \gtrsim 10^{-5}\sigma_1$.

Do **not** use it when the small directions are the answer. Whitening divides by them, ridge filters
them, and a condition estimate is entirely about them. In each of those the SVD route is required,
and lesson 94's section 6 measured what happens to a whitener that trusts its smallest direction.

Two situations reliably break the condition and are common: features on wildly different scales, and
one feature that is nearly a linear combination of others, which is what happens whenever a
categorical variable is one-hot encoded without dropping a level.

### 5.2 Truncation against ridge

Both are filters on the singular directions. Their filter factors are

$$
f_i^{\text{trunc}} = \begin{cases}1 & i \le k\\ 0 & i > k\end{cases} ,
\qquad
f_i^{\text{ridge}} = \frac{\sigma_i^2}{\sigma_i^2+\lambda} .
$$

Truncation is a step; ridge is a smooth sigmoid in $\log\sigma$ with its half point at
$\sigma = \sqrt\lambda$.

**Where the difference does not matter.** When the spectrum has a clear gap, both filters cut in the
gap, both give nearly the same answer, and the choice is arbitrary. Lesson 98's section 6 measured a
version of this: on a blur, the two *penalty operators* differed by only $1.037$ because the operator
had already sorted its own directions.

**Where it does matter, three cases.**

*A spectrum with no gap*, which is the common case for a power-law spectrum. Truncation has to
choose an integer $k$, and the answer jumps as $k$ changes by one. Ridge moves continuously, so a
continuous parameter search is possible and the L-curve or GCV can differentiate the curve. This is
the main practical argument for ridge.

*Directions near the cutoff.* Truncation either keeps a direction at full strength or discards it
entirely. Ridge keeps a direction near $\sigma^2 = \lambda$ at half strength, which is the correct
thing to do when the signal and the noise in that direction are comparable. In mean squared error
terms, exercise 2.4's per-component optimum is $f_i = \sigma_i^2\alpha_i^2/(\sigma_i^2\alpha_i^2 +
\tau^2)$, a smooth function, and the ridge filter is a one-parameter family that can match it while
the step filter cannot.

*Interpretability, where truncation wins.* A truncated solution lives in an explicitly named
$k$-dimensional subspace, so it can be reported as "the answer is a combination of these $k$
patterns". A ridge solution has support in every direction, so it cannot.

**A useful equivalence.** If the spectrum is roughly geometric, $\sigma_i \approx \sigma_1 r^i$, then
choosing $\lambda$ is choosing $k \approx \log_r(\sqrt\lambda/\sigma_1)$, and the two methods differ
only in how sharply they treat the two or three directions nearest the cut. That is why they usually
give errors within a small factor of each other, which lesson 98's section 6 measured directly.

### 5.3 What a low-rank adapter cannot do

Take the exact statement from 1.5: after $k$ minibatch steps of size $B$, the accumulated update has
rank at most $\min(kB, d)$, and lesson 94 measured that bound attained.

**The situation in which a rank-$r$ adapter must lose information is therefore any run with
$kB > r$**, which is every run of practical length. That sounds fatal and is not, and the reason is
worth being precise about: the bound says the update *can* reach rank $kB$, and the measurement in
exercise 3.4 says the update's **spectrum decays** even when its rank is full. So the adapter loses
the tail, not everything past $r$.

**What the loss looks like.** Write the full update's SVD as $\Delta = \sum_i s_i u_i v_i^{\mathsf T}$.
The best rank-$r$ adapter keeps the leading $r$ terms, so the residual is $\sum_{i>r} s_i u_i
v_i^{\mathsf T}$, with Frobenius norm $\sqrt{\sum_{i>r}s_i^2}$. In terms of behaviour, the discarded
directions are the ones the optimizer visited least, which are typically the ones learned latest.
So the failure mode is not a uniform degradation; it is the loss of whatever the last part of
training was doing.

**Three concrete situations where the loss bites.**

*Long training on a task far from the base model.* The further the target is from the start, the more
directions the path has to cover, and the more of them a fixed $r$ drops. Adapters work best for
adaptation, which is what they are named for.

*Large batches.* A batch of $1024$ makes each single gradient rank $1024$, so even one step exceeds
any practical $r$. The rank bound is a statement about $kB$, and $B$ is under the practitioner's
control.

*Tasks needing a genuinely new direction.* If the base model's weight matrix has no component along
some direction the task needs, and the adapter's rank is spent on directions that improve the loss
faster, the needed direction never gets represented.

**How to detect it.** Fit the full fine-tuning update once, take its singular values, and read off
the captured fraction as a function of $r$, exactly as exercise 3.4 does. If the curve is still
rising steeply at the $r$ you were going to use, the adapter is too small, and the measurement takes
one SVD.

---

## Lesson 95, Optimization for Machine Learning

### 1.1 The gradient flow and its discretization

The **gradient flow** is the ordinary differential equation

$$
\frac{dx}{dt} = -\nabla f(x) , \qquad x(0) = x_0 .
$$

Its solution moves downhill at every instant, and $f(x(t))$ is non-increasing because
$\frac{d}{dt}f(x(t)) = \nabla f \cdot \dot x = -\lVert\nabla f\rVert^2 \le 0$.

Apply **explicit Euler** from lesson 67 with step $h$:

$$
x_{k+1} = x_k + h\,\dot x(x_k) = x_k - h\,\nabla f(x_k) ,
$$

which is gradient descent with learning rate $h$. The identification is exact, not an analogy: the
learning rate **is** the step size of a first order ODE solver.

Three consequences follow at once, and each is a Part 10 result wearing an optimization name.

- The trajectory error is $O(h)$ per unit time, so halving the learning rate halves the distance from
  the continuous path. Lesson 95's section 2 fitted that at $0.9984$.
- There is an absolute stability limit, exercise 1.2.
- Everything the ODE literature knows about stiffness applies, and a loss with a wide spectrum of
  curvatures is a stiff problem.

### 1.2 The step size limit from absolute stability

Near a minimum $x^\star$, expand $\nabla f(x) \approx H(x - x^\star)$ with $H$ the Hessian,
symmetric positive semidefinite with eigenvalues $L_1 \ge \dots \ge L_n \ge 0$. Writing
$e_k = x_k - x^\star$, the gradient descent iteration becomes

$$
e_{k+1} = (I - hH)\,e_k .
$$

Diagonalize: in the eigenbasis each component evolves independently as
$e_{k+1}^{(i)} = (1 - hL_i)e_k^{(i)}$. The iteration is a linear recurrence whose amplification
factor in direction $i$ is $|1 - hL_i|$, and it converges if and only if

$$
|1 - hL_i| < 1 \quad\text{for every } i
\quad\Longleftrightarrow\quad
0 < h < \frac{2}{L_i} \quad\text{for every } i
\quad\Longleftrightarrow\quad
h < \frac{2}{L_{\max}} .
$$

**This is exactly lesson 74's absolute stability condition** $|1 + h\lambda| \le 1$ applied to the
linearized flow, whose eigenvalues are $\lambda_i = -L_i$. Nothing about optimization enters. The
region of absolute stability for Euler's method is the disc $|1 + h\lambda| \le 1$ centred at $-1$,
and $h < 2/L_{\max}$ is the statement that every $-hL_i$ lands inside it.

Lesson 95's section 3 located the boundary by bisecting on the amplification factor and matched
$2/L$ to $1.8\times10^{-12}$. The table there shows the amplification crossing $1$ exactly at the
predicted step, and one per cent past it the iterate grows by $16$ in $200$ steps.

**Note what sets the limit.** $L_{\max}$, the largest curvature, not the average one and not the
condition number. One stiff direction caps the step for all of them, which is the definition of
stiffness from lesson 74.

### 1.3 Heavy ball as a second order equation

The damped oscillator is

$$
\ddot x + a\,\dot x + \nabla f(x) = 0 .
$$

Discretize with central differences for $\ddot x$ and a backward difference for $\dot x$, at step
$h$:

$$
\frac{x_{k+1} - 2x_k + x_{k-1}}{h^2} + a\,\frac{x_k - x_{k-1}}{h} + \nabla f(x_k) = 0 .
$$

Multiply by $h^2$ and collect:

$$
x_{k+1} = x_k - h^2\,\nabla f(x_k) + (1 - ah)\,(x_k - x_{k-1}) ,
$$

which is heavy ball, $x_{k+1} = x_k - \alpha\nabla f(x_k) + \beta(x_k - x_{k-1})$, with

$$
\alpha = h^2 , \qquad \beta = 1 - a h .
$$

**So $\alpha$ and $\beta$ are not independent.** They are a step size and a damping constant in
disguise, and refining the discretization means moving both: halving $h$ divides $\alpha$ by $4$ and
moves $\beta$ towards $1$. Lesson 95's section 4 measured the resulting trajectory against the
continuous one at order $0.9995$.

Two readings that fall out of this.

**Why $\beta = 0.9$ is a default.** Inverting, $h = (1-\beta)/a$ and $\alpha = h^2$, so a small $h$
means a $\beta$ near $1$. Common defaults correspond to a modest damping and a small step.

**Why the two have to be retuned together.** Changing $\beta$ alone changes both the damping and,
through $\alpha = h^2$, the effective step, so a grid search over $(\alpha,\beta)$ is searching a
skewed parametrization of $(h, a)$.

### 1.4 The variance of a minibatch gradient

Let $g_1,\dots,g_N$ be the per-sample gradients with mean $\bar g$ and population covariance
$\Sigma$. Draw $B$ indices **without replacement** and average. This is sampling from a finite
population, so

$$
\operatorname{Cov}\!\left[\hat g_B\right]
= \frac{\Sigma}{B}\left(1 - \frac{B-1}{N-1}\right) ,
$$

and taking the trace gives the expected squared error

$$
\mathbb{E}\lVert \hat g_B - \bar g\rVert^2
= \frac{\operatorname{tr}\Sigma}{B}\left(1 - \frac{B-1}{N-1}\right) .
$$

The bracket is the **finite population correction**. Without it the formula is the familiar $1/B$,
which is what sampling **with** replacement gives.

**At $B = N$** the bracket is $1 - (N-1)/(N-1) = 0$, so the variance is exactly zero. That is not an
approximation and not a limit: taking every sample once gives $\bar g$ itself, which is the full
batch gradient. Lesson 95's section 6 measured the formula to $2.4$ per cent and the full batch row
at $2\times10^{-33}$, which is rounding.

**Why the correction matters here and not in lesson 92.** Monte Carlo integration samples from an
infinite population, so the correction is absent. Machine learning batches are often a real fraction
of the dataset: at $B = N/2$ the correction is $\approx 1/2$, so the noise is half what the plain
$1/B$ law predicts, and any argument built on $1/B$ alone is off by a factor of two.

### 1.5 Two reasons a run leaves a sharp minimum

**Reason one, the step size, and it needs no noise.** Section 1.2's condition is local, so a minimum
of curvature $c$ is an attracting fixed point of gradient descent only while $h < 2/c$. Two minima
with different curvatures have different limits, so there is a range of $h$ in which the flat one is
a fixed point and the sharp one is not. Lesson 95's section 8 measured both limits at $2/c$ to
within $0.03$ per cent, on wells of curvature $3.25$ and $27.8$, giving a window a factor of $8.5$
wide.

**Reason two, the minibatch noise, and it needs the first reason to have happened.** Inside that
window a noiseless run still cannot leave: it oscillates around the sharp minimum without escaping,
and lesson 95's section 8 located the ejection threshold at $3.25$ times the destabilization one.
Between those two thresholds the sampling noise decides, and section 9 measured it: at $2.5$ times
the limit, a batch of $1$ leaves in $100$ per cent of runs and every larger batch in $0$ per cent.

**Which one needs noise:** only the second. And the order matters. Below the stability window the
noise does nothing at all, which the first row of section 9's table shows as zeros across every
batch size. **The step size opens the door and the batch size walks through it.**

That is a narrower statement than the usual "SGD noise finds flat minima", and it is the one the
measurement supports.

### 2.1 The optimal step and the contraction factor

On a quadratic with Hessian eigenvalues in $[m, L]$, the error obeys
$e_{k+1} = (I - hH)e_k$, so the worst case contraction per step is

$$
\rho(h) = \max_{L_i \in [m, L]} |1 - hL_i| = \max\left(|1 - hm|,\ |1 - hL|\right) ,
$$

because $|1 - h\mu|$ is linear in $\mu$ and therefore maximized at an endpoint. For $h > 0$ the
first term decreases in $h$ while the second increases past $h = 1/L$, so the minimum of the maximum
is where they are equal:

$$
1 - hm = hL - 1 \quad\Longrightarrow\quad h^\star = \frac{2}{L + m} ,
$$

giving

$$
\rho^\star = 1 - \frac{2m}{L+m} = \frac{L - m}{L + m} = \frac{\kappa - 1}{\kappa + 1} ,
\qquad \kappa = L/m .
$$

To reach a relative error $\tau$ takes $k$ steps with $\rho^{\star k} \le \tau$, that is

$$
k \ge \frac{\log(1/\tau)}{\log(1/\rho^\star)}
\approx \frac{\kappa}{2}\log\frac{1}{\tau} ,
$$

using $\log(1/\rho^\star) = -\log\left(1 - \frac{2}{\kappa+1}\right) \approx 2/\kappa$ for large
$\kappa$.

Lesson 95's section 5 measured that at $17$, $89$, $886$, $8856$ and $88556$ iterations against
predictions of $18$, $92$, $921$, $9210$ and $92103$, a worst ratio of $1.084$ over four decades of
$\kappa$.

**And the whole thing disappears under preconditioning.** With $H^{-1}$ as the preconditioner the
effective condition number is $1$, the contraction factor is $0$, and the count is $1$. That is the
same section's last column, at a saving of $88556$.

### 2.2 Sharpness is the trace of the Hessian

At a minimum $x^\star$ of a quadratic, $f(x^\star + \delta) - f(x^\star) = \tfrac12\delta^{\mathsf T}H\delta$.
Take $\delta = r u$ with $u$ uniform on the unit sphere in $\mathbb{R}^d$. Then

$$
\mathbb{E}\left[f(x^\star + ru) - f(x^\star)\right]
= \frac{r^2}{2}\,\mathbb{E}\left[u^{\mathsf T}Hu\right] .
$$

Now $\mathbb{E}[uu^{\mathsf T}] = I/d$ by symmetry: the expectation is a symmetric matrix invariant
under every rotation, hence a multiple of $I$, and its trace is $\mathbb{E}\lVert u\rVert^2 = 1$. So

$$
\mathbb{E}\left[u^{\mathsf T}Hu\right] = \operatorname{tr}\left(H\,\mathbb{E}[uu^{\mathsf T}]\right)
= \frac{\operatorname{tr}H}{d} ,
$$

and

$$
\boxed{\ \mathbb{E}\left[f(x^\star + ru) - f(x^\star)\right] = \frac{r^2}{2}\cdot\frac{\operatorname{tr}H}{d}\ }
$$

The **worst** direction gives $\tfrac12 r^2 L_{\max}$, attained along the top eigenvector.

**So "flat" and "sharp" are two different numbers.** The average damage is the trace, the worst case
is the largest eigenvalue, and their ratio is $d\,L_{\max}/\operatorname{tr}H$, which is $1$ on an
isotropic problem and large on an ill-conditioned one. Lesson 95's section 10 measured the trace
prediction to $1.9$ per cent and the worst-case-over-average ratio at $4.48$ for $\kappa = 100$.

The practical reading: a sharpness measure that samples random directions is measuring
$\operatorname{tr}H$, and one that maximizes over directions is measuring $L_{\max}$. They can rank
two minima differently, so any claim that flatness predicts generalization has to say which one it
means.

### 2.3 The natural gradient is invariant

Let $x = Ay$ with $A$ invertible, and write $\ell(y) = L(Ay)$. Then

$$
\nabla_y \ell = A^{\mathsf T}\nabla_x L , \qquad
F_y = A^{\mathsf T}F_x A ,
$$

the first by the chain rule and the second because the Fisher matrix is a covariance of gradients,
$F = \mathbb{E}[\nabla L\,\nabla L^{\mathsf T}]$, so it transforms as $A^{\mathsf T}(\cdot)A$.

The natural gradient step in $y$ coordinates is

$$
\Delta y = -\eta\,F_y^{-1}\nabla_y\ell
= -\eta\,(A^{\mathsf T}F_xA)^{-1}A^{\mathsf T}\nabla_xL
= -\eta\,A^{-1}F_x^{-1}A^{-\mathsf T}A^{\mathsf T}\nabla_xL
= -\eta\,A^{-1}F_x^{-1}\nabla_xL .
$$

Mapping back to $x$ coordinates, $\Delta x = A\,\Delta y = -\eta F_x^{-1}\nabla_x L$, which is the
step the method would have taken without the change of variables. **The invariance is exact, at every
step size, for every invertible $A$**, and by induction the whole trajectory is invariant.

**Where the argument needs $F$ to transform as it does.** The cancellation is
$(A^{\mathsf T}F A)^{-1}A^{\mathsf T} = A^{-1}F^{-1}$, and it needs the preconditioner to pick up
$A^{\mathsf T}(\cdot)A$ exactly. Any $F$ built as a covariance or an expectation of second
derivatives does. A preconditioner that does not, for example a diagonal approximation, breaks the
identity: the diagonal of $A^{\mathsf T}FA$ is not $A^{\mathsf T}\operatorname{diag}(F)A$. So Adam is
invariant only under **diagonal** rescalings, not general ones, which is exercise 5.3's subject.

Plain gradient descent has $\Delta y = -\eta A^{\mathsf T}\nabla_x L$ and $\Delta x =
-\eta AA^{\mathsf T}\nabla_x L$, which equals $-\eta\nabla_x L$ only if $A$ is orthogonal. Lesson
95's section 12 measured the natural gradient's drift at $2.8\times10^{-16}$ and plain descent's at
$0.47$.

### 2.4 The Fisher information of a Gaussian model

Take $y \sim \mathcal{N}(m(\theta), \sigma^2 I)$ with $n$ observations. The log likelihood is

$$
\ell(\theta) = -\frac{1}{2\sigma^2}\lVert y - m(\theta)\rVert^2 + \text{const} ,
$$

so with $J = \partial m/\partial\theta$,

$$
\nabla_\theta \ell = \frac{1}{\sigma^2}J^{\mathsf T}(y - m(\theta)) .
$$

The Fisher information is the covariance of the score at the true parameter:

$$
F = \mathbb{E}\left[\nabla\ell\,\nabla\ell^{\mathsf T}\right]
= \frac{1}{\sigma^4}J^{\mathsf T}\,\mathbb{E}\left[(y-m)(y-m)^{\mathsf T}\right]J
= \frac{1}{\sigma^4}J^{\mathsf T}(\sigma^2 I)J
= \frac{J^{\mathsf T}J}{\sigma^2} .
$$

With $\sigma = 1$ that is $J^{\mathsf T}J$ exactly, which is the **Gauss-Newton** matrix for the
least squares objective $\tfrac12\lVert y - m(\theta)\rVert^2$.

The true Hessian of that objective is

$$
\nabla^2 = J^{\mathsf T}J - \sum_i r_i\,\nabla^2 m_i , \qquad r = y - m(\theta) ,
$$

so **all three agree when the residual is zero and only then.** Lesson 95's section 12 measured
$F = J^{\mathsf T}J$ to exactly $0$, the Hessian gap to $0$ at a zero residual, and $1.33$ of the
norm at a large one, where the Hessian is no longer positive definite.

That last fact is the practical value of Gauss-Newton: it is positive semidefinite by construction,
because it is $J^{\mathsf T}J$, so it can be inverted or shifted safely where the Hessian cannot. It
achieves that by dropping exactly the term that can be negative.

### 2.5 The rank of a sampled curvature

For a least squares loss on a batch of $B$ samples with design rows $a_1,\dots,a_B$,

$$
\hat H = \frac{1}{B}\sum_{j=1}^{B} a_j a_j^{\mathsf T} = \frac{1}{B}A_{\text{batch}}^{\mathsf T}A_{\text{batch}} .
$$

By exercise 2.5 of lesson 94, a sum of $B$ outer products has rank at most $B$, so

$$
\operatorname{rank}(\hat H) \le \min(B, d) .
$$

The same holds for the Gauss-Newton matrix of any model, $\hat H = \frac1B J_{\text{batch}}^{\mathsf T}J_{\text{batch}}$
with $J_{\text{batch}}$ of shape $B \times d$.

**What that implies for a Newton step.** If $B < d$ then $\hat H$ is singular, exactly and not
approximately, so $\hat H^{-1}g$ does not exist. A shift $\hat H + \mu I$ makes it exist and the step
is then $O(1/\mu)$ in the $d - B$ directions the batch never saw, which is a large step in a
direction about which there is no information.

If $B \ge d$ then $\hat H$ is generically invertible, and the problem changes rather than
disappearing: its smallest eigenvalues are estimated from $B$ samples with a relative error of order
$\sqrt{d/B}$, and the Newton step divides by them. Lesson 95's section 11 measured both regimes: at
$B = 8$ with $d = 20$ the rank is $8$ and the run diverges; at $B = 32$ the matrix is invertible and
Newton is still $6.43$ times worse than plain descent.

### 3.1 Nesterov momentum and the ODE it discretizes

Nesterov's method evaluates the gradient at a look-ahead point:

$$
y_k = x_k + \beta(x_k - x_{k-1}) , \qquad
x_{k+1} = y_k - \alpha\nabla f(y_k) .
$$

Eliminating $y$ and expanding $\nabla f(y_k) \approx \nabla f(x_k) + \beta\nabla^2 f\,(x_k -
x_{k-1})$ shows Nesterov is heavy ball plus a term proportional to the Hessian times the velocity,
which is a **gradient correction**. In the continuous limit with $\alpha = h^2$ and
$\beta = 1 - ah$, that extra term is $O(h^3)$, so Nesterov discretizes the **same** damped
oscillator to the same order, with a different local truncation error.

```python
import numpy as np
from nalib import mlopt


def nesterov(problem, start, step, momentum, iterations):
    x = np.array(start, dtype=float)
    previous = x.copy()
    for _ in range(iterations):
        look = x + momentum * (x - previous)
        nxt = look - step * problem["gradient"](look)
        previous, x = x, nxt
    return x


problem = mlopt.quadratic(np.array([1.0, 3.0, 8.0]), seed=42)
start = np.ones(problem["dimension"])
horizon, damping = 4.0, 1.5
exact = mlopt.damped_flow(problem, start, horizon, damping, steps=65536)["x"]

print(f"{'steps':>8}{'h':>12}{'heavy ball':>16}{'nesterov':>16}{'ratio':>9}")
counts = (256, 512, 1024, 2048, 4096)
heavy_errors, nesterov_errors = [], []
for steps in counts:
    h = horizon / steps
    heavy = mlopt.heavy_ball(problem, start, h * h, 1.0 - damping * h, steps)["x"]
    fast = nesterov(problem, start, h * h, 1.0 - damping * h, steps)
    heavy_errors.append(float(np.linalg.norm(heavy - exact)))
    nesterov_errors.append(float(np.linalg.norm(fast - exact)))
    print(f"{steps:>8}{h:>12.6f}{heavy_errors[-1]:>16.6e}{nesterov_errors[-1]:>16.6e}"
          f"{heavy_errors[-1] / nesterov_errors[-1]:>9.4f}")

steps_used = np.log([horizon / s for s in counts])
print(f"\nheavy ball order : {np.polyfit(steps_used, np.log(heavy_errors), 1)[0]:.4f}")
print(f"nesterov order   : {np.polyfit(steps_used, np.log(nesterov_errors), 1)[0]:.4f}")
```

**Both are first order against the same ODE**, at fitted orders of $0.9995$ and $0.9852$, which is
the claim. The two curves are parallel on a log-log plot, so the look-ahead changes the constant and
not the order.

**And it changes the constant in the wrong direction.** The ratio settles at $0.433$, so Nesterov's
iterate is $2.3$ times **further** from the continuous trajectory than heavy ball's, at every step
size tested. As an ODE integrator the look-ahead is a worse choice.

That is not a contradiction of Nesterov's reputation; it is a clarification of what the reputation
is about. The famous $O(1/k^2)$ rate is a statement about the **optimization** error with a
$k$-dependent $\beta_k = (k-1)/(k+2)$, and it comes from an estimate-sequence argument about how
fast $f(x_k)$ falls. It says nothing about how closely the iterates track any continuous curve, and
this measurement says they track it less closely. **Being a better optimizer and being a better
integrator are different properties, and the measurement separates them.**

### 3.2 A diagonal preconditioner, and how much of $\kappa$ it removes

Adam without momentum divides each coordinate by the square root of a running average of squared
gradients. On a quadratic, that estimates $\sqrt{\operatorname{diag}(H^2)}$ near the solution rather
than $\operatorname{diag}(H)$, and the preconditioner it builds is diagonal, so it can only remove
the part of the conditioning that is visible in the coordinate basis.

```python
import numpy as np
from nalib import mlopt


def diagonal_preconditioner(problem, start, step, iterations, decay=0.999, guard=1e-12):
    """Accumulate squared gradients and precondition by their root, which is Adam's second moment."""
    x = np.array(start, dtype=float)
    second = np.zeros_like(x)
    for k in range(1, iterations + 1):
        g = problem["gradient"](x)
        second = decay * second + (1.0 - decay) * g * g
        scale = np.sqrt(second / (1.0 - decay ** k)) + guard
        x = x - step * g / scale
    return {"x": x, "scale": scale}


print("an exact diagonal preconditioner, on a diagonal Hessian and on a rotated one")
print(f"{'rotated':>9}{'kappa(H)':>12}{'kappa after diag(H)':>22}")
for rotated in (False, True):
    curvatures = np.geomspace(1.0, 200.0, 10)
    hessian = (mlopt.quadratic(curvatures, seed=4)["hessian"] if rotated
               else np.diag(curvatures))
    diagonal = np.diag(1.0 / np.sqrt(np.diag(hessian)))
    print(f"{str(rotated):>9}{np.linalg.cond(hessian):>12.2f}"
          f"{np.linalg.cond(diagonal @ hessian @ diagonal):>22.2f}")

print("\nand what Adam's running second moment actually converges to")
problem = mlopt.least_squares_model(512, 10, noise=0.4, condition=8.0, seed=15)
curvature = np.diag(problem["hessian"])
rng = np.random.default_rng(3)
x = problem["solution"] + 0.5 * np.ones(problem["parameters"])
second = np.zeros(problem["parameters"])
decay, batch = 0.99, 16
for k in range(1, 4001):
    rows = rng.choice(problem["samples"], size=batch, replace=False)
    g = problem["gradient"](x, rows)
    second = decay * second + (1.0 - decay) * g * g
    x = x - 0.02 * g / (np.sqrt(second / (1.0 - decay ** k)) + 1e-8)

draws = 400
noise_scale = np.zeros(problem["parameters"])
for _ in range(draws):
    rows = rng.choice(problem["samples"], size=batch, replace=False)
    noise_scale += (problem["gradient"](x, rows) - problem["gradient"](x)) ** 2
noise_scale /= draws

print(f"{'i':>4}{'diag(H)':>14}{'Adam second moment':>22}{'minibatch noise':>19}")
for i in range(problem["parameters"]):
    print(f"{i:>4}{curvature[i]:>14.6f}{second[i]:>22.6f}{noise_scale[i]:>19.6f}")
print(f"\ncorrelation with diag(H)        : "
      f"{np.corrcoef(second, curvature)[0, 1]:>8.4f}")
print(f"correlation with the noise scale: "
      f"{np.corrcoef(second, noise_scale)[0, 1]:>8.4f}")
print(f"second moment / diag(H), spread : "
      f"{np.max(second / curvature) / np.min(second / curvature):>8.4f}")

print("\nthe same test with heteroscedastic noise, which breaks Fisher = Gauss-Newton")
spread = np.exp(np.linspace(-2.0, 2.0, base_rows := problem["samples"]))
rng2 = np.random.default_rng(9)
design = problem["design"]
dirty = design @ problem["truth"] + spread * rng2.standard_normal(base_rows)
hessian = design.T @ design / base_rows
solution = np.linalg.lstsq(design, dirty, rcond=None)[0]


def dirty_gradient(v, rows=None, a=design, b=dirty):
    block = a if rows is None else a[rows]
    wanted = b if rows is None else b[rows]
    return block.T @ (block @ np.asarray(v, dtype=float) - wanted) / block.shape[0]


y = solution + 0.5 * np.ones(problem["parameters"])
second2 = np.zeros(problem["parameters"])
for k in range(1, 4001):
    rows = rng2.choice(base_rows, size=batch, replace=False)
    g = dirty_gradient(y, rows)
    second2 = decay * second2 + (1.0 - decay) * g * g
    y = y - 0.02 * g / (np.sqrt(second2 / (1.0 - decay ** k)) + 1e-8)
noise2 = np.zeros(problem["parameters"])
for _ in range(draws):
    rows = rng2.choice(base_rows, size=batch, replace=False)
    noise2 += (dirty_gradient(y, rows) - dirty_gradient(y)) ** 2
noise2 /= draws
print(f"correlation with diag(H)        : "
      f"{np.corrcoef(second2, np.diag(hessian))[0, 1]:>8.4f}")
print(f"correlation with the noise scale: "
      f"{np.corrcoef(second2, noise2)[0, 1]:>8.4f}")
```

**Two separate findings, and the second one contradicts the premise of the exercise.**

*The first table.* A diagonal preconditioner removes the conditioning that is diagonal. On a
diagonal Hessian it takes $\kappa = 200$ to exactly $1$. Rotate the same spectrum into a generic
basis and it takes $200$ to $129$, because the ill-conditioning has moved into the off-diagonal
entries where a diagonal scaling cannot reach it. **That is the honest limit of every diagonal
method**, Adam included.

*The second table, which did not answer the question it was asked, and the reason is the
interesting part.* The exercise's premise is that Adam's second moment estimates the curvature. The
first test cannot check that, because **the second moment correlates with $\operatorname{diag}(H)$
at $0.995$ and with the minibatch noise scale at $0.986$**, and the ratio of the second moment to
$\operatorname{diag}(H)$ varies by only $1.53$ across coordinates.

Both correlations are high because on a correctly specified least squares model with homoscedastic
noise, the gradient noise covariance **is** proportional to the Hessian. That is exercise 2.4's
identity, $F = J^{\mathsf T}J/\sigma^2$, read the other way: the Fisher information and the
Gauss-Newton matrix coincide, so the two candidate explanations of what Adam is estimating are the
same matrix and no correlation test can separate them.

The third block breaks the tie by breaking the assumption. Give each sample its own noise scale, so
the noise covariance is $J^{\mathsf T}\Sigma J$ with $\Sigma \neq \sigma^2 I$ and no longer
proportional to $J^{\mathsf T}J$, and the two correlations separate: $0.952$ against
$\operatorname{diag}(H)$ and $0.991$ against the noise. **The second moment follows the noise**,
which is what it is defined to be: a running average of squared gradients, and near a minimum a
gradient is mostly noise.

The separation is real and it is not dramatic, and that is worth saying rather than dressing up.
Even a misspecified least squares model has a noise covariance built from the same Jacobian, so the
two matrices stay correlated however the per-sample scales are chosen. A clean separation needs a
model whose curvature and whose noise come from different places, which a least squares problem
cannot provide.

So the answer to "how much of the condition number does Adam remove" is: **that is the wrong
question about the wrong matrix.** Adam is dividing by an estimate of the gradient noise scale per
coordinate, which is a normalization rather than a preconditioning. On a deterministic quadratic the
quantity degenerates entirely, because $g \to 0$ and the second moment goes to zero with it.

This is exercise 5.3's subject, and the measurement here is the evidence for the claim made there:
Adam is a normalized first order method, not a compromised second order one.

### 3.3 Stochastic Newton with a Levenberg-Marquardt shift

Section 11 measured Newton diverging at a batch of $8$ with $20$ parameters. Add a shift
$\hat H + \mu I$ and sweep $\mu$.

```python
import math
import numpy as np
from nalib import mlopt

problem = mlopt.least_squares_model(512, 20, noise=0.2, condition=50.0, seed=42)
start = problem["solution"] + np.ones(problem["parameters"])
floor = problem["loss"](problem["solution"])
plain_step = 0.5 * 2.0 / problem["largest"]
updates, batch = 200, 8

reference = None
rng = np.random.default_rng(43)
x = np.array(start, dtype=float)
picks = [rng.choice(problem["samples"], size=batch, replace=False) for _ in range(updates)]
for rows in picks:
    x = x - plain_step * problem["gradient"](x, rows)
reference = problem["loss"](x) - floor

print(f"gradient descent at batch {batch}: excess loss {reference:.6e}")
print(f"the sampled curvature has rank {np.linalg.matrix_rank(problem['curvature'](picks[0]))} "
      f"for {problem['parameters']} parameters")
print(f"the full Hessian's eigenvalues run from {problem['smallest']:.3e} "
      f"to {problem['largest']:.3f}")
print(f"\n{'shift':>10}{'excess loss':>16}{'against sgd':>14}")
crossover = None
for power in range(-8, 3):
    shift = 10.0 ** power
    y = np.array(start, dtype=float)
    for rows in picks:
        with np.errstate(over="ignore", invalid="ignore"):
            y = y - mlopt.newton_step(problem, y, rows, shift=shift)
        if not np.all(np.isfinite(y)):
            break
    value = problem["loss"](y) - floor if np.all(np.isfinite(y)) else math.inf
    shown = "diverged" if not np.isfinite(value) else f"{value:.6e}"
    ratio = "-" if not np.isfinite(value) else f"{value / reference:.4f}"
    if np.isfinite(value) and value < reference and crossover is None:
        crossover = shift
    print(f"{shift:>10.0e}{shown:>16}{ratio:>14}")
print(f"\nthe smallest shift that beats plain descent is {crossover:.0e}"
      if crossover else "\nno shift beat plain descent")
```

**The shift has to reach $\mu = 1$ before the method beats plain descent, and the full Hessian's
largest eigenvalue is $1.000$.** So the shift that works is the one comparable with $L_{\max}$, at
which point $(\hat H + \mu I)^{-1} \approx \mu^{-1}I$ and the step is essentially $-g/\mu$: gradient
descent with learning rate $1$.

**The shift that makes stochastic Newton work is the shift that turns it back into gradient
descent**, and the window is one decade wide: at $\mu = 10^{-1}$ it is still $1.13$ times worse, and
at $\mu = 10$ it is $4.6$ times worse for the opposite reason, because the step has become too
short. The best row, $\mu = 1$, beats plain descent by $1.17$, which is not much of a prize for a
$d \times d$ solve per step.

This is why Levenberg-Marquardt is a **trust region** method in disguise rather than a fix for noise:
the shift is chosen to control the step length, not to denoise the curvature, and controlling the
step length is what gradient descent already does.

### 3.4 Sharpness-aware minimization, measured

SAM takes the gradient at a perturbed point, $\nabla f(x + \rho\, g/\lVert g\rVert)$, which
approximately descends the worst-case loss in a ball. Does it land somewhere flatter?

```python
import numpy as np
from nalib import mlopt


def sam(problem, start, step, radius, batch, updates, seed=0):
    """Sharpness-aware minimization: step using the gradient at the worst nearby point."""
    x = np.array(start, dtype=float)
    rng = np.random.default_rng(seed)
    for _ in range(updates):
        rows = rng.choice(problem["samples"], size=batch, replace=False)
        g = problem["gradient"](x, rows)
        norm = float(np.linalg.norm(g))
        moved = x + radius * g / norm if norm > 0 else x
        x = x - step * problem["gradient"](moved, rows)
    return x


def plain(problem, start, step, batch, updates, seed=0):
    x = np.array(start, dtype=float)
    rng = np.random.default_rng(seed)
    for _ in range(updates):
        rows = rng.choice(problem["samples"], size=batch, replace=False)
        x = x - step * problem["gradient"](x, rows)
    return x


problem = mlopt.two_wells(samples=400, seed=42)
start = problem["sharp_at"]
step = 0.5 * 2.0 / problem["sharp_curvature"]
grid = problem["grid"]
scale = grid[1] - grid[0]


def curvature_at(point):
    return float((problem["value"](point + scale) - 2.0 * problem["value"](point)
                  + problem["value"](point - scale)) / scale ** 2)


trials, updates = 60, 1500
print(f"the two wells sit at {problem['flat_at']:.2f} and {problem['sharp_at']:.2f}, "
      f"curvatures {problem['flat_curvature']:.2f} and {problem['sharp_curvature']:.2f}")
print(f"\n{'method':>12}{'radius':>9}{'ended flat':>13}{'mean curvature':>17}")
for name, radius in (("plain", 0.0), ("sam", 0.02), ("sam", 0.08), ("sam", 0.20)):
    flat = 0
    curvatures = []
    for trial in range(trials):
        if name == "plain":
            end = plain(problem, start, step, 8, updates, seed=trial)
        else:
            end = sam(problem, start, step, radius, 8, updates, seed=trial)
        end = float(end)
        if abs(end - problem["flat_at"]) < abs(end - problem["sharp_at"]):
            flat += 1
        curvatures.append(curvature_at(end))
    print(f"{name:>12}{radius:>9.2f}{flat / trials:>13.3f}{np.mean(curvatures):>17.4f}")
```

**The two columns disagree, and that is the result.**

**SAM never moves a run to the other well.** The third column is $0.000$ at every radius, including
the largest. The run starts in the sharp well and stays in the sharp well, so on this landscape SAM
does not do the thing it is usually described as doing.

**And it does flatten the neighbourhood it settles in.** The fourth column falls monotonically with
the radius, from $27.24$ at radius $0$ to $21.35$ at radius $0.20$, a reduction of $22$ per cent. SAM
is pushing the iterate to the flatter side of the well it is in, which is a real effect and a much
smaller one than escaping.

**Why both are true.** SAM perturbs by $\rho\,g/\lVert g\rVert$, which near a minimum is a small
step in the direction the gradient happens to point, and it descends the perturbed gradient. That
biases the endpoint towards lower curvature **within** a basin. It does not add energy, so it cannot
carry a run over a barrier, which is what changing basins requires.

The measurement is worth having because **flatness has an operational definition**, exercise 2.2's
trace, so the claim can be checked rather than argued. A SAM result reported without the fourth
column is reporting a hope, and one reported without the third is claiming more than it measured.

### 3.5 Newton-CG with a matrix-free Hessian-vector product

Newton-CG never forms $H$. It needs only $Hv$, which finite differences of the gradient supply at the
cost of one extra gradient evaluation:

$$
Hv \approx \frac{\nabla f(x + \epsilon v) - \nabla f(x)}{\epsilon} ,
\qquad \epsilon = \sqrt{\varepsilon}\,\frac{1 + \lVert x\rVert}{\lVert v\rVert} .
$$

```python
import numpy as np
from nalib import mlopt


def hessian_vector(problem, x, v, epsilon=None):
    """One Hessian-vector product from two gradients, with lesson 68's step."""
    scale = float(np.linalg.norm(v))
    if scale == 0.0:
        return np.zeros_like(v)
    step = (np.sqrt(np.finfo(float).eps) * (1.0 + float(np.linalg.norm(x))) / scale
            if epsilon is None else epsilon)
    return (problem["gradient"](x + step * v) - problem["gradient"](x)) / step


def newton_cg(problem, start, outer=8, inner=40, tolerance=1e-10):
    """Newton's method with the linear system solved by conjugate gradient, matrix free."""
    x = np.array(start, dtype=float)
    products = 0
    for _ in range(outer):
        g = problem["gradient"](x)
        if float(np.linalg.norm(g)) < tolerance:
            break
        direction = np.zeros_like(x)
        residual = -g
        search = residual.copy()
        squared = float(residual @ residual)
        for _ in range(inner):
            moved = hessian_vector(problem, x, search)
            products += 1
            denominator = float(search @ moved)
            if denominator <= 0.0:
                break
            step = squared / denominator
            direction = direction + step * search
            residual = residual - step * moved
            updated = float(residual @ residual)
            if np.sqrt(updated) < 1e-8 * float(np.linalg.norm(g)):
                break
            search = residual + (updated / squared) * search
            squared = updated
        x = x + direction
    return {"x": x, "products": products}


for dimension in (20, 100, 400):
    problem = mlopt.quadratic(np.geomspace(1.0, 400.0, dimension), seed=9)
    start = np.ones(dimension)
    out = newton_cg(problem, start)
    dense = dimension * dimension
    print(f"d = {dimension:>4}: final norm {float(np.linalg.norm(out['x'])):.3e}, "
          f"{out['products']} products, "
          f"memory {4 * dimension} against {dense} for the dense Hessian, "
          f"a factor of {dense / (4 * dimension):.1f}")
```

**The memory saving is $d/4$**, because Newton-CG holds four vectors and a dense Hessian holds $d^2$
entries. At $d = 400$ that is a factor of $100$, and at the parameter counts of a real model it is
the difference between possible and impossible.

The accuracy is the other half. The finite difference product carries lesson 68's error, about
$\sqrt\varepsilon$ relative, so the Newton direction is only accurate to $10^{-8}$. That is enough:
the outer iteration is self-correcting, and an inexact Newton step still converges superlinearly if
the inner tolerance is tightened as the outer residual falls. Lesson 97's exercise 3.5 replaces the
finite difference with an exact forward-over-reverse product and removes even that error.

### 4.1 Does the largest stable learning rate move during training

The edge of stability question: on a nonquadratic loss the curvature changes as the iterate moves, so
$2/L_{\max}$ is a moving target.

```python
import numpy as np
from nalib import mlopt

problem = mlopt.two_wells(samples=400, seed=42)
grid = problem["grid"]
scale = grid[1] - grid[0]


def curvature_at(point):
    return float((problem["value"](point + scale) - 2.0 * problem["value"](point)
                  + problem["value"](point - scale)) / scale ** 2)


start = -2.0
checkpoints = (0, 5, 20, 80, 320, 1280)
for name, step in (("well inside the limit", 0.9 * 2.0 / problem["sharp_curvature"]),
                   ("just inside the flat well's limit",
                    0.98 * 2.0 / problem["flat_curvature"]),
                   ("just past it", 1.02 * 2.0 / problem["flat_curvature"])):
    x = start
    seen = 0
    print(f"\na full batch run from x = {start}, {name}, step {step:.6f}")
    print(f"{'iteration':>11}{'x':>12}{'loss':>12}{'local curvature':>18}"
          f"{'2 / curvature':>16}{'step / limit':>15}")
    for target in checkpoints:
        while seen < target:
            x = x - step * problem["gradient"](x)
            seen += 1
        c = curvature_at(x)
        limit = 2.0 / c if c > 0 else np.inf
        shown = "inf" if not np.isfinite(limit) else f"{limit:.6f}"
        ratio = 0.0 if not np.isfinite(limit) else step / limit
        print(f"{seen:>11}{x:>12.6f}{problem['value'](x):>12.6f}{c:>18.4f}"
              f"{shown:>16}{ratio:>15.4f}")
    tail = []
    for _ in range(200):
        x = x - step * problem["gradient"](x)
        tail.append(float(x))
    swing = max(tail) - min(tail)
    print(f"  over the last 200 steps the iterate swings by {swing:.6e}")
```

**The curvature the run sees is not the curvature at its destination.** At $x = -2$ the local
curvature is **negative**, $-1.44$, because the starting point is outside both wells on a
concave stretch, so $2/c$ is not even defined there. The run then descends into the flat well and
the curvature settles at $3.2531$.

**The first run never approaches its stability limit**, stopping at a ratio of $0.105$, and its
swing over the last $200$ steps is exactly $0$. It converges to a point.

**The second run sits at a ratio of $0.98$ and still converges**, to a swing of $5.3\times10^{-15}$,
which is rounding. That is the correction the measurement makes to the obvious guess: being *near*
the limit is not enough. The amplification is $|1 - 0.98\cdot 2| = 0.96 < 1$, so the transient decays,
slowly, and $0.96^{1280}$ is $10^{-23}$.

**Only the third run, at $1.02$ times the limit, does not settle.** There the amplification is
$|1 - 1.02\cdot2| = 1.04 > 1$, the linearized iteration grows, and it stops growing only when the
nonlinearity of the well takes over. It ends in a **period two orbit** with a swing of $0.220$ that
does not decay, around a minimum it never reaches. **That is the edge of stability, and the boundary
between the second run and the third is exactly $h = 2/c$.**

The fourth and fifth columns of that run show the self-limiting mechanism directly. The oscillation
carries the iterate out to where the curvature is lower, $3.063$ instead of $3.253$, so the ratio
$h/(2/c)$ falls back to $0.960$ and the growth stops. **The run has moved itself to the edge and
stayed there**, which is exactly the feedback loop exercise 5.1 describes, visible on a
one-dimensional loss.

On a real loss the curvature also **rises** as the loss falls, so a run at a fixed $h$ moves towards
that boundary rather than away from it, and settles on it. Exercise 5.1 develops the argument.

### 4.2 The linear scaling rule under preconditioning

Section 7 located the rule's breaking point at the batch where the step crosses $2/L$. If that is
right, preconditioning the problem should move the breaking point by exactly the factor by which it
moves $L$.

```python
import numpy as np
from nalib import mlopt

base_step, epochs, noise = 0.02, 40, 0.3


def rescaled(matrix, target, factor):
    """Wrap a design matrix as a problem, so the same sweep can run on any rescaling of it."""
    design = matrix * factor
    hessian = design.T @ design / design.shape[0]
    solution = np.linalg.lstsq(design, target, rcond=None)[0]

    def loss(v, rows=None, a=design, b=target):
        block = a if rows is None else a[rows]
        wanted = b if rows is None else b[rows]
        residual = block @ np.asarray(v, dtype=float) - wanted
        return float(residual @ residual / (2 * residual.size))

    def gradient(v, rows=None, a=design, b=target):
        block = a if rows is None else a[rows]
        wanted = b if rows is None else b[rows]
        return block.T @ (block @ np.asarray(v, dtype=float) - wanted) / block.shape[0]

    return {"design": design, "target": target, "samples": design.shape[0],
            "parameters": design.shape[1], "hessian": hessian, "solution": solution,
            "largest": float(np.linalg.eigvalsh(hessian)[-1]),
            "loss": loss, "gradient": gradient}


base = mlopt.least_squares_model(512, 12, noise=noise, condition=20.0, seed=42)
print(f"{'scaling':>10}{'L':>10}{'2/L':>10}{'predicted break':>18}{'measured break':>17}"
      f"{'ratio':>9}")
for factor in (0.5, 1.0, 2.0, 3.0):
    problem = rescaled(base["design"], base["target"], factor)
    limit = 2.0 / problem["largest"]
    predicted = limit / base_step
    floor = problem["loss"](problem["solution"])
    start = problem["solution"] + np.ones(problem["parameters"])
    measured = None
    for power in range(0, 12):
        size = 2 ** power
        if size > problem["samples"]:
            break
        with np.errstate(over="ignore", invalid="ignore"):
            run = mlopt.minibatch_descent(problem, start, base_step * size, size,
                                          max(epochs * problem["samples"] // size, 1), seed=7)
            excess = (problem["loss"](run["x"]) - floor
                      if np.all(np.isfinite(run["x"])) else np.inf)
        if not np.isfinite(excess) or excess > 1.0:
            measured = size
            break
    ratio = "-" if measured is None else f"{measured / predicted:.3f}"
    print(f"{factor:>10.2f}{problem['largest']:>10.4f}{limit:>10.4f}"
          f"{predicted:>18.1f}{str(measured):>17}{ratio:>9}")
```

**Rescaling the design by $c$ scales $L$ by $c^2$ and the predicted break by $1/c^2$**, and the
measured break is the first power of two past it. The last column is the test: if the explanation in
section 7 is right, that ratio should sit between $1$ and $2$ at every scaling, because the sweep can
only report powers of two.

A note on how this measurement had to be set up. The first version of it preconditioned the problem
by $H^{-1/2}$ and compared against the original, and it measured nothing, because
`least_squares_model` normalizes its design so that $L$ is already exactly $1$. Preconditioning a
problem whose Hessian is the identity is a no-op, and the two rows of the table were identical.
**Rescaling by an explicit factor is the same test without the accident.**

The practical reading is uncomfortable and correct: **the batch size at which the linear scaling rule
fails is a property of the loss surface, not of the data or the noise**, so it moves whenever
anything changes the curvature, which normalization layers do continuously during training.

### 4.3 The escape rate against the step size

Kramers' picture says the escape rate from a well over a barrier $\Delta$ with noise scale $D$ goes
like $\exp(-\Delta/D)$. For SGD near a minimum the stationary distribution is approximately
$\exp(-2f/(h\sigma^2))$, so the exponent should be $-2\Delta/(h\sigma^2)$ and the log escape rate
should be linear in $1/h$.

```python
import math
import numpy as np
from nalib import mlopt

problem = mlopt.two_wells(samples=400, seed=42)
base = 2.0 / problem["sharp_curvature"]
barrier = problem["barrier"] - problem["sharp_value"]
per_sample = np.array([problem["gradient"](problem["sharp_at"], np.array([i]))
                       for i in range(problem["samples"])])
sigma_squared = float(np.var(per_sample))
print(f"barrier height {barrier:.6f}, per sample gradient variance {sigma_squared:.4f}")

trials, updates, batch = 60, 3000, 1
print(f"\n{'h / (2/c)':>11}{'h':>10}{'escaped':>10}{'rate per step':>16}"
      f"{'predicted exponent':>21}")
rows = []
for factor in (1.6, 1.8, 2.0, 2.2, 2.4):
    step = factor * base
    escaped = 0
    for trial in range(trials):
        out = mlopt.minibatch_descent(problem, problem["sharp_at"], step, batch, updates,
                                      seed=200 + trial)
        if abs(float(out["x"]) - problem["flat_at"]) < abs(float(out["x"])
                                                           - problem["sharp_at"]):
            escaped += 1
    share = escaped / trials
    rate = -math.log(max(1.0 - share, 1.0 / (2 * trials))) / updates
    exponent = -2.0 * barrier / (step * sigma_squared)
    rows.append((step, rate, exponent))
    print(f"{factor:>11.2f}{step:>10.5f}{share:>10.3f}{rate:>16.3e}{exponent:>21.4f}")

usable = [(s, r, e) for s, r, e in rows if r > 0]
if len(usable) >= 3:
    slope = np.polyfit([1.0 / s for s, _, _ in usable],
                       [math.log(r) for _, r, _ in usable], 1)[0]
    print(f"\nfitted slope of log(rate) against 1/h : {slope:.4f}")
    print(f"predicted -2 * barrier / sigma^2       : "
          f"{-2.0 * barrier / sigma_squared:.4f}")
```

**The fit rejects the Kramers picture.** The measured slope of $\log(\text{rate})$ against $1/h$ is
$-4.68$, and the thermal prediction $-2\Delta/\sigma^2$ is $-0.745$. Those differ by a factor of
$6.3$, so the escape here is **not** simple thermal activation over a barrier.

The escape counts say why. From a step ratio of $1.6$ to $2.4$ the escaped fraction goes
$0.000$, $0.000$, $0.017$, $0.100$, $0.983$: that is not the smooth exponential a Kramers rate
predicts, it is a threshold with a soft edge. And $2.4$ is well inside lesson 95's section 8 range,
where a noiseless run oscillates with a swing that grows with $h$. **What the noise is doing is not
carrying the iterate over a barrier from a resting state; it is perturbing an already growing
oscillation into crossing one.**

Two mechanisms, and the measurement separates them because they predict different shapes. A thermal
rate is a straight line in $\log(\text{rate})$ against $1/h$ with a computable slope. An instability
assisted crossing is a sharp rise located by the deterministic swing. The data is the second, which
is the honest answer and the opposite of what the exercise's phrasing expects.

The Kramers picture is not wrong in general; it is the right picture when $h$ is small enough that
the deterministic dynamics is contracting. That regime is the first two rows here, and in it the
escape rate is not zero but too small to measure in $3000$ steps, which is exactly what a rate of
$e^{-5}$ per correlation time would give.

### 5.1 The edge of stability

Section 3 says a minimum of curvature $L$ is attracting only while $h < 2/L$. Section 4.1 says the
curvature a run sees changes as it moves. Put those together on a real loss, where the curvature
generally **grows** as the loss falls, and here is what has to happen.

**The argument.** Start with a small $h$ and an initialization in a flat region, so $hL_{\max} \ll 2$
and the run converges quickly in every direction. As the loss falls, the run enters a region of
higher curvature, and $L_{\max}$ rises. Once $hL_{\max}$ passes $2$, the top eigendirection stops
contracting and starts oscillating with growing amplitude. That oscillation carries the run back out
to lower curvature, which brings $hL_{\max}$ back below $2$.

The result is a **feedback loop that pins $L_{\max}$ at approximately $2/h$**, and the run sits there
for the rest of training: not converged, because the top direction is marginally unstable, and not
diverged, because the instability is self-limiting.

**What this predicts, and each is checkable.**

- $L_{\max}$ measured during training should rise, then flatten at $2/h$, then track $2/h$ if $h$ is
  changed. Halving the learning rate should double the plateau.
- The loss should fall non-monotonically, in small oscillations superimposed on a downward trend,
  because the top direction is oscillating.
- Progress continues, because the other $d-1$ directions have $hL_i < 2$ and are converging normally.
  Only one direction is unstable.
- A learning rate schedule that decays $h$ moves the plateau up, which is a different reason for
  decaying it than the usual noise argument.

**Why this is not a pathology.** Section 8's measurement separates two thresholds: losing a minimum
at $2/c$ and leaving it at $3.25$ times that. The edge of stability sits between them, which is
exactly the regime where the run is neither converging nor escaping. It is a stable operating point
of the discretization, and it exists because gradient descent is a numerical method with a stability
boundary, not because the loss surface is strange.

### 5.2 Why flat minima might generalize

Exercise 2.2 gives the exact statement: the expected loss increase under a random perturbation of
length $r$ is $r^2\operatorname{tr}H/(2d)$. That is a fact about the training loss and its Hessian,
and it is fully numerical.

**The argument connecting it to generalization has three steps.**

*Step one, numerical.* A flat minimum is one where $\operatorname{tr}H$ is small, so the training
loss changes little under a perturbation of the parameters. This is exercise 2.2, exact for a
quadratic and a good approximation in a small ball.

*Step two, an assumption.* The test loss is the training loss of a different sample, and one may
model that difference as a perturbation of the loss surface, hence approximately a perturbation of
the location of the minimum. If the minimum moves by $\delta$ and the surface is flat, the loss at
the old point is still near optimal.

*Step three, numerical again.* Therefore the gap between training and test loss is bounded by
something like $\lVert\delta\rVert^2\operatorname{tr}H/(2d)$, small when the minimum is flat.

**Step two is the one that is not numerical**, and it is where the argument can fail. It assumes the
difference between the training and test surfaces acts like a translation, and there is no reason it
should. Two specific failures:

*Reparametrization.* Flatness is not invariant. Rescaling a layer's weights by $c$ and the next
layer's by $1/c$ leaves the function unchanged and multiplies parts of the Hessian by $c^{\pm2}$, so
the same function can be made arbitrarily flat or sharp without changing anything a test set could
detect. Any flatness measure that is not reparametrization invariant is measuring the coordinates.

*Which flatness.* Exercise 2.2 gives two numbers, the trace and the largest eigenvalue, measured by
lesson 95's section 10 to differ by $4.48$ at $\kappa = 100$. A claim about flatness has to say which
one it means, and different papers mean different ones.

**So the honest position:** step one and step three are theorems, step two is an empirical
regularity that holds often and not always, and the counterexamples are constructions that exploit
exactly its non-invariance.

### 5.3 What Adam actually approximates

Adam's update, ignoring the bias corrections and the momentum, is

$$
x_{k+1} = x_k - \frac{\eta}{\sqrt{v_k} + \epsilon}\,g_k ,
\qquad
v_k = \text{running average of } g_k^2 \text{ (elementwise)} .
$$

**What matrix is being estimated.** The elementwise average of $g^2$ estimates
$\operatorname{diag}\left(\mathbb{E}[gg^{\mathsf T}]\right)$, the diagonal of the **uncentred second
moment** of the gradient. Near a minimum, where $\mathbb{E}[g] \approx 0$, that is the diagonal of
the gradient covariance, which is the diagonal of the **empirical Fisher** matrix. And exercise 2.4
showed that for a Gaussian model the Fisher is $J^{\mathsf T}J$, the Gauss-Newton matrix. So Adam is
approximating $\operatorname{diag}(F)$, a diagonal approximation to the Gauss-Newton matrix.

**Why the square root.** This is the part that is usually stated as a heuristic and is not. A natural
gradient step would use $F^{-1}$, so a diagonal version would use $1/\operatorname{diag}(F)$, not
$1/\sqrt{\operatorname{diag}(F)}$. Three reasons the square root is used anyway:

*Units.* Dividing by $F$ makes the step scale-free in the parameters, which means the learning rate
$\eta$ has no meaning and the step length is entirely determined by the curvature estimate. Dividing
by $\sqrt F$ leaves the step proportional to $g/\sqrt{\mathbb{E}[g^2]}$, which is a **normalized
gradient**, of size about $1$ per coordinate. So $\eta$ becomes an interpretable step length in
parameter space, which is what makes Adam's learning rate transferable between problems.

*Noise.* $v$ is a noisy estimate. Dividing by $v$ amplifies its noise linearly; dividing by
$\sqrt v$ amplifies it by half as much. Given section 11's finding that inverting a noisy curvature
estimate is the failure mode of second order methods, taking a square root is a deliberate
softening.

*It is not trying to be Newton.* $\mathbb{E}[g^2]$ near a minimum is $\approx \operatorname{diag}(F)$
times the gradient noise scale, so the quantity Adam divides by mixes curvature with noise. Treating
it as a curvature estimate and inverting it fully would be using it for something it is not.

**How it avoids section 11's failure.** Three ways, and all three matter.

- It is **diagonal**, so there is no matrix to invert and no small eigenvalue of a near-singular
  matrix to divide by. The rank bound of exercise 2.5 does not apply, because no matrix is being
  inverted.
- The **square root** halves the amplification of the estimate's noise.
- The $\epsilon$ in the denominator is a hard floor on the divisor, which is exactly the
  Levenberg-Marquardt shift of exercise 3.3, applied where it is cheap.

So Adam is not a compromised second order method. It is a **normalized** first order method whose
normalizer happens to resemble a curvature estimate, and every one of its departures from the second
order ideal is a departure that section 11 measured the need for.

---

## Lesson 96, Floating Point in Deep Learning

### 1.1 The two splits, and what each one breaks

Both formats use $16$ bits: one sign, some exponent, the rest mantissa.

| | exponent | stored mantissa | unit roundoff | largest | smallest positive |
|---|---|---|---|---|---|
| `float16` | $5$ | $10$ | $4.88\times10^{-4}$ | $65504$ | $5.96\times10^{-8}$ |
| `bfloat16` | $8$ | $7$ | $3.91\times10^{-3}$ | $3.39\times10^{38}$ | $9.18\times10^{-41}$ |

**`float16` is prone to range failures**, at both ends. Overflow, because $\exp$ leaves it at
$11.09$ and a softmax computes $\exp$ of raw scores. Underflow, because gradients are routinely below
$10^{-7}$ and $6\times10^{-8}$ is the smallest number it has. Neither is a loss of accuracy: an
overflowed value is `inf` and an underflowed one is exactly $0$.

**`bfloat16` is prone to precision failures.** It carries $2.1$ decimal digits, so any accumulation
stagnates quickly: a running sum of ones stops at $256$, which is narrower than a single layer. Its
failures are gradual rather than catastrophic, which is a large part of why it is preferred.

The two are the same $16$ bits, so the trade is exact: `float16` has $8$ times the precision and
`bfloat16` has $5.2\times10^{33}$ times the range.

### 1.2 Why $\exp$ overflows at $\log$ of the largest value

A format can represent numbers up to some $M$. $\exp$ is increasing and continuous, so
$e^x \le M \iff x \le \log M$. Above that the true value has no representation and the result is
`inf`. There is nothing about $\exp$ in this beyond its being increasing; the same argument gives
the overflow input for any monotone function.

For `float16`, $M = 65504$ and

$$
\log M = \log 65504 = 11.0899\ldots
$$

Lesson 96's section 2 bisected for the actual threshold and found $11.0898$, matching to $2$ parts in
$10^6$. For `bfloat16` and `float32`, $M \approx 3.4\times10^{38}$ and $\log M = 88.72$, the same
number for both because they share an exponent field.

**Why $11.09$ is small.** It is not small as a floating point number; it is small as a **logit**. A
model with any confidence produces scores in the tens, so the direct softmax formula fails on
ordinary inputs rather than on extreme ones. That is the whole practical content of the number.

### 1.3 Why the softmax shift costs nothing

Most numerical fixes trade something. The stable quadratic formula of lesson 05 costs an extra
operation and a branch. The augmented ridge form of lesson 94 costs a taller matrix. Iterative
refinement costs an extra solve.

The softmax shift costs a maximum and a subtraction, and gives up **nothing**, for three reasons.

**It is an identity, not an approximation.** Exercise 2.5 proves
$\text{softmax}(x) = \text{softmax}(x - c)$ exactly, for any $c$, in exact arithmetic. So the shifted
form computes the same function rather than a nearby one.

**It cannot lose accuracy.** After the shift every argument to $\exp$ is $\le 0$, so every
$e^{x_i - m} \in (0, 1]$ and the largest is exactly $1$. Summing numbers that are all at most $1$,
with the largest present, is the best conditioned version of that sum: the ratio of the largest term
to the total is at least $1/n$, so there is no cancellation and no growth.

**It cannot overflow, for any input.** $e^{\text{something} \le 0} \le 1$, always.

The one thing it can lose is in representing $x - m$ itself when $\lVert x\rVert$ is enormous, which
lesson 96's section 3 measured at $6.6\times10^{-12}$ for an offset of $10^6$ and bounded by
$\lVert x\rVert u$. That is the rounding of the input, not of the method, and no formulation can
avoid it.

**So the shifted form should be the only way anyone writes a softmax**, and the same for logsumexp.

### 1.4 Why loss scaling does not round

Let $S = 2^p$. Multiplying a floating point number by a power of two changes only its exponent
field: if $x = m \cdot 2^e$ with $m$ the stored mantissa, then $2^p x = m \cdot 2^{e+p}$, the same
mantissa with a different exponent. **The mantissa is unchanged, so no rounding occurs**, and the same
argument applies to the division by $S$ afterwards.

Two conditions, both usually satisfied:

- The result must not overflow or underflow. $e + p$ has to stay inside the exponent range. That is
  exactly the window lesson 96's section 5 measured, $2^{10}$ to $2^{40}$ for `float16` with
  gradients near $10^{-8}$.
- It must not become subnormal, where the mantissa is no longer left-justified and bits are lost.

So loss scaling is a **change of units**. The gradient is computed in units of $1/S$ and converted
back, and the only thing that happens numerically is the cast to $16$ bits in between. That cast is
the one operation that rounds, and the whole purpose of the scale is to put the value where that cast
does the least damage.

**Why a power of two specifically.** Any $S$ works mathematically. A non-power of two rounds twice,
once on the multiply and once on the divide, adding $2u$ of relative error for nothing. Frameworks
that adjust the scale dynamically always double or halve it for this reason.

### 1.5 Where a `float16` sum of ones stops

$$
\boxed{\ 2048\ }
$$

**Derivation.** Let $u$ be the unit roundoff, $2^{-11}$ for `float16`. The running total $T$ grows by
$1$ each step. Adding $1$ to $T$ gives a real number $T + 1$, which is rounded to the nearest
representable value. The gap between representable numbers near $T$ is $\text{ulp}(T) \approx 2uT$
for a $T$ in the binade, so the addition survives while

$$
1 > \frac{\text{ulp}(T)}{2} = uT
\quad\Longleftrightarrow\quad
T < \frac{1}{u} = 2^{11} = 2048 .
$$

At $T = 2048$ exactly, the numbers representable near $2048$ are spaced by $2$, so $2048 + 1 = 2049$
rounds to nearest and ties to even, giving $2048$. The sum stops.

Lesson 96's section 6 measured exactly $2048$ for `float16` and exactly $256$ for `bfloat16`, both
$1/u$ to the digit, and `float32`'s predicted $1.68\times10^7$ was not reached in $200000$ terms.

**The `bfloat16` number is the one to remember.** A `bfloat16` accumulator cannot count past $256$,
which is narrower than any layer in any modern network. That single integer is why mixed precision
exists.

### 2.1 The largest representable value

An IEEE format with $e$ exponent bits and $t$ stored mantissa bits uses a bias $B = 2^{e-1} - 1$.
The stored exponent field ranges over $0$ to $2^e - 1$, with $0$ reserved for zero and subnormals and
$2^e - 1$ reserved for infinities and NaNs. So the largest usable stored exponent is $2^e - 2$,
giving an unbiased exponent

$$
E_{\max} = (2^e - 2) - B = 2^e - 2 - 2^{e-1} + 1 = 2^{e-1} - 1 = B .
$$

The largest mantissa is $1.\underbrace{11\cdots1}_{t} = 2 - 2^{-t}$. Therefore

$$
\boxed{\ M = (2 - 2^{-t})\,2^{B} , \qquad B = 2^{e-1} - 1 . \ }
$$

Check it on the three cases:

- `float16`: $e = 5$, $t = 10$, $B = 15$, $M = (2 - 2^{-10})\cdot 2^{15} = 65504$. Exact.
- `float32`: $e = 8$, $t = 23$, $B = 127$, $M = 3.4028\times10^{38}$.
- `bfloat16`: $e = 8$, $t = 7$, $B = 127$, $M = (2 - 2^{-7})\cdot2^{127} = 3.3895\times10^{38}$,
  slightly below `float32`'s because the mantissa is shorter.

The smallest **normal** number is $2^{1-B}$, the smallest stored exponent above the reserved zero,
with mantissa $1.0$. The smallest **subnormal** is that times $2^{-t}$.

### 2.2 The stagnation point, and what happens if the terms shrink

Let the running total be $T$ and the term be $a$. In a format with unit roundoff $u$, the addition
$T \oplus a$ returns the nearest representable value to $T + a$. Representable values near $T$ are
spaced by $\text{ulp}(T)$, and $u = \text{ulp}(T)/(2T)$ by definition, so the spacing is $2uT$. The
addition changes $T$ only if $a$ is at least half the spacing:

$$
a > uT \quad\Longleftrightarrow\quad T < \frac{a}{u} .
$$

So a sum of **equal** terms of size $a$ stagnates once $T$ reaches $a/u$, after $1/u$ terms
regardless of $a$, which is exercise 1.5.

**If the terms shrink, the answer changes, and how it changes decides everything.** Suppose
$a_k = k^{-p}$ and let $T_k$ be the partial sum.

- **$p > 1$:** the series converges to $T_\infty$, and the terms fall while the total does not, so
  stagnation is certain and early. It happens at $k^{-p} \approx uT_\infty$, that is
  $k \approx (uT_\infty)^{-1/p}$. In `float16` with $T_\infty = 2$ and $p = 2$ that is $k \approx 32$.
- **$p = 1$:** the harmonic series. $T_k \approx \log k$ and $a_k = 1/k$, so stagnation is at
  $1/k = u\log k$, roughly $k \approx 1/(u\log(1/u))$. Slower than the equal-term case by a
  logarithm.
- **$p < 1$:** $T_k \approx k^{1-p}/(1-p)$ and $a_k = k^{-p}$, so the condition $a_k > uT_k$ becomes
  $k^{-p} > u k^{1-p}/(1-p)$, that is $k < (1-p)/u$. **The same $1/u$ scaling as equal terms**, with a
  constant.

The general rule: **stagnation happens when the terms fall below $u$ times the total**, and whether
that is soon or late depends on how fast the total grows relative to the terms. A convergent series
always stagnates; a divergent one stagnates at $O(1/u)$ terms.

### 2.3 The $\sqrt n\,u$ estimate, and where $nu$ comes from

Sequential summation computes $T_k = \text{fl}(T_{k-1} + a_k)$, so each step introduces a relative
error $\delta_k$ with $|\delta_k| \le u$:

$$
T_k = (T_{k-1} + a_k)(1 + \delta_k) .
$$

Unrolling, the computed sum is

$$
\hat T_n = \sum_{k=1}^{n} a_k \prod_{j=k}^{n}(1 + \delta_j)
\approx \sum_k a_k\left(1 + \sum_{j \ge k}\delta_j\right)
= T_n + \sum_k a_k\,E_k ,
\qquad E_k = \sum_{j\ge k}\delta_j .
$$

**The worst case.** If every $\delta_j = u$ with the same sign, then $|E_k| \le (n - k + 1)u \le nu$
and

$$
|\hat T_n - T_n| \le nu\sum_k |a_k| ,
$$

which is the standard $nu$ bound. It is attained only when all the roundings conspire, which needs
the terms to have a very particular structure.

**The random sign case.** Model the $\delta_j$ as independent, mean zero, with standard deviation
$u/\sqrt3$ (uniform on $[-u,u]$). Then $E_k$ is a sum of $n-k+1$ of them, so
$\operatorname{sd}(E_k) \approx u\sqrt{n-k+1}/\sqrt3$, and the total error is a sum of $n$ terms with
independent-ish random signs, giving

$$
\operatorname{sd}(\hat T_n - T_n) \sim u\sqrt{n}\,\lVert a\rVert
$$

up to a constant. **So the exponent is $1/2$, not $1$**, on generic data, and lesson 96's section 7
fitted it at $0.538$.

The same measurement found the $\sqrt n\,u$ **bound** loose by $555$, which is not a contradiction:
the bound uses $\sum|a_k|$ while the realized error uses $\lVert a\rVert_2$, and for $n$ random
numbers those differ by $\sqrt n$. Two different quantities, both correctly labelled.

### 2.4 Pairwise summation and the measured $0.109$

Pairwise summation splits the array in half, sums each half recursively, and adds the two results.
Let $\varepsilon(n)$ bound the error for $n$ terms. The recursion is

$$
\varepsilon(n) \le 2\,\varepsilon(n/2) + u\,\lvert T\rvert ,
$$

one rounding for the final addition. With $\varepsilon(1) = 0$ and $d = \log_2 n$ levels, the error
accumulates one rounding per level along any path from a leaf to the root, so

$$
|\hat T_n - T_n| \le u\log_2(n)\sum_k|a_k| ,
$$

which is $O(u\log n)$ against sequential's $O(un)$.

**Why the measured exponent is $0.109$ and not $0$.** The prediction $u\log_2 n$ is not constant in
$n$; it grows logarithmically. On a log-log plot, $\log n$ appears as a very slowly rising line
whose local slope is $1/\ln n$, which over $n$ from $10^3$ to $10^6$ is between $0.14$ and $0.07$.
The measured $0.109$ sits in that range.

And the random-sign refinement applies here too: the errors along different paths are independent,
so the realized spread grows like $\sqrt{\log n}$, whose local log-log slope is half of the above,
$0.07$ to $0.035$. The measurement, $0.109$, is between the two, which is what a mixture of the
worst case and the random case looks like.

**Either way the conclusion is unchanged.** Sequential grows like $n^{0.538}$ and pairwise like
$n^{0.109}$, so at a million terms pairwise is $100$ times tighter, as lesson 96 measured.

### 2.5 Softmax invariance and the range after shifting

**Invariance.** For any constant $c$,

$$
\text{softmax}(x - c)_i
= \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}
= \frac{e^{-c}e^{x_i}}{e^{-c}\sum_j e^{x_j}}
= \frac{e^{x_i}}{\sum_j e^{x_j}}
= \text{softmax}(x)_i .
$$

The factor $e^{-c}$ appears in every term of the numerator and the denominator and cancels exactly.
Nothing is approximated, and the identity holds for every $c$, not just the maximum.

**The range after shifting by the maximum.** Take $c = m = \max_j x_j$. For every $i$,

$$
x_{\min} - m \le x_i - m \le 0 ,
$$

so, applying the increasing function $\exp$,

$$
e^{x_{\min} - x_{\max}} \le e^{x_i - m} \le e^0 = 1 ,
$$

with the upper bound attained by whichever $i$ achieves the maximum. **So every exponential is in
$(0, 1]$ and at least one equals exactly $1$.**

Three consequences, and each is a separate good property:

- **No overflow is possible**, for any input whatsoever, because $1$ is representable in every format.
- **The sum is between $1$ and $n$**, so it cannot overflow either, and the division is by a number
  of order $1$.
- **The sum is well conditioned.** Its largest term is $1$ and its total is at least $1$, so the
  condition number $\sum|a_i|/|\sum a_i|$ is exactly $1$: there is no cancellation, because every
  term is positive.

Underflow is still possible, for terms with $x_i \ll m$, and it is harmless: those terms are
negligible in the sum and their softmax probabilities are correctly $0$ to the format's precision.

### 3.1 The two `float8` splits

E4M3 has $4$ exponent bits and $3$ mantissa bits; E5M2 has $5$ and $2$. Both are used in practice,
E4M3 for forward activations and E5M2 for gradients, and the reason is exactly the range against
precision trade of section 1.

```python
import math
import numpy as np
from nalib import mixedprecision as mp


def describe_small(exponent_bits, mantissa_bits, name):
    """Section 1's table for any split, using exercise 2.1's formulas."""
    bias = 2 ** (exponent_bits - 1) - 1
    largest = (2.0 - 2.0 ** (-mantissa_bits)) * 2.0 ** bias
    smallest_normal = 2.0 ** (1 - bias)
    return {"name": name, "bits": 1 + exponent_bits + mantissa_bits,
            "exponent_bits": exponent_bits, "mantissa_bits": mantissa_bits,
            "unit_roundoff": 2.0 ** (-mantissa_bits - 1), "largest": largest,
            "smallest_normal": smallest_normal,
            "smallest_subnormal": smallest_normal * 2.0 ** (-mantissa_bits),
            "decimal_digits": mantissa_bits * math.log10(2.0),
            "overflow_input_for_exp": math.log(largest)}


rows = [describe_small(4, 3, "e4m3"), describe_small(5, 2, "e5m2")]
rows += [mp.describe(name) for name in ("float16", "bfloat16", "float32")]
print(f"{'format':>10}{'bits':>6}{'exp':>5}{'mant':>6}{'unit roundoff':>16}"
      f"{'largest':>13}{'smallest':>13}{'digits':>8}{'exp breaks at':>16}")
for row in rows:
    print(f"{row['name']:>10}{row['bits']:>6}{row['exponent_bits']:>5}"
          f"{row['mantissa_bits']:>6}{row['unit_roundoff']:>16.3e}{row['largest']:>13.3e}"
          f"{row['smallest_subnormal']:>13.3e}{row['decimal_digits']:>8.2f}"
          f"{row['overflow_input_for_exp']:>16.3f}")

print(f"\ne4m3 against e5m2: {rows[1]['unit_roundoff'] / rows[0]['unit_roundoff']:.1f} "
      f"times the precision, {rows[0]['largest'] / rows[1]['largest']:.4f} of the range")
print(f"a float16 sum of ones stops at {1 / mp.describe('float16')['unit_roundoff']:.0f}, "
      f"an e4m3 one at {1 / rows[0]['unit_roundoff']:.0f}, "
      f"an e5m2 one at {1 / rows[1]['unit_roundoff']:.0f}")
```

**The numbers explain the convention.** E4M3 has exactly twice the precision of E5M2 and
$0.0042$ of its range, a factor of $239$, so it is used where values are near $1$ and precision
matters, which is activations. E5M2 keeps the range, reaching $5.7\times10^{4}$ against E4M3's
$240$, so it is used for gradients, which span decades.

Both carry **less than one decimal digit**: $0.90$ and $0.60$. That is the number to sit with. An
E5M2 value is specified to within a factor of $1.25$, so these formats are not approximations of a
number, they are a bucket the number fell into.

The last line is the warning. **An E5M2 running sum stops at $8$**, and an E4M3 one at $16$. No
accumulation of any kind can happen in these formats. They are storage formats and nothing else, and
every `float8` matrix multiply on real hardware accumulates in `float32`, exactly as section 9
argued for `float16` and more so.

### 3.2 Dynamic loss scaling

The standard implementation: start high, halve the scale and skip the step whenever an `inf` or
`nan` appears in the scaled gradient, and double it after a run of clean steps.

```python
import numpy as np
from nalib import mixedprecision as mp


def dynamic_scaling(gradients, start_power=24, patience=200, name="float16"):
    """Skip and halve on overflow, double after a clean run. Returns the history."""
    scale = 2.0 ** start_power
    clean = 0
    skipped = 0
    applied = 0
    powers = []
    for gradient in gradients:
        out = mp.loss_scaled_gradient(gradient, scale, name)
        if out["overflowed"]:
            scale = max(scale / 2.0, 1.0)
            clean = 0
            skipped += 1
        else:
            applied += 1
            clean += 1
            if clean >= patience:
                scale = scale * 2.0
                clean = 0
        powers.append(np.log2(scale))
    return {"skipped": skipped, "applied": applied, "powers": powers,
            "final_power": np.log2(scale)}


rng = np.random.default_rng(0)
steps, size = 2000, 512
schedule = np.geomspace(1e-3, 1e-8, steps)
stream = [level * rng.standard_normal(size) for level in schedule]

print(f"a run of {steps} steps whose gradients shrink from 1e-3 to 1e-8")
print(f"{'start power':>13}{'skipped':>10}{'skip rate':>12}{'final power':>14}"
      f"{'zeros at the end':>19}")
for start in (8, 16, 24, 32, 40):
    out = dynamic_scaling(stream, start_power=start)
    final = mp.loss_scaled_gradient(stream[-1], 2.0 ** out["final_power"], "float16")
    print(f"{start:>13}{out['skipped']:>10}{out['skipped'] / steps:>12.4f}"
          f"{out['final_power']:>14.0f}{final['zeros']:>19.6f}")
```

**The controller reaches a working scale from every starting point, and the cost of starting high is
a rounding error.** Starting at $2^{32}$ or $2^{40}$ costs $8$ and $16$ skipped steps out of $2000$,
a skip rate of $0.4$ and $0.8$ per cent, after which the scale has halved into the window. Starting
at $2^8$ costs no skips at all and climbs by doubling.

**But look at where they end.** The three low starts finish at $2^{18}$, $2^{26}$ and $2^{34}$, and
the two high ones both at $2^{33}$. The controller does not converge to a single scale: with a
patience of $200$ it can only double ten times in $2000$ steps, so a run that starts low is still
climbing when the run ends. Every one of them ends with no zeros in the gradient, so all five work,
and they work at scales that differ by $2^{16}$.

**That looseness is the design, and the asymmetry behind it is the point.** Overflow is loud, an
`inf` that is trivial to detect. Underflow is silent, a zero that looks like a converged coordinate.
So the algorithm is built to climb until it hits the ceiling and back off, and it does not need to
find the optimum because the window measured in section 5 is $30$ powers of two wide.

### 3.3 Stochastic rounding

Round up with probability equal to the fractional position between the two neighbours, down
otherwise. The expected value is then exact, so errors have mean zero and a long sum accumulates
$\sqrt n$ instead of stagnating.

```python
import numpy as np
from nalib import mixedprecision as mp


def stochastic_round(value, name, rng):
    """Round to one of the two neighbours, with probability proportional to the distance."""
    below = float(np.asarray(mp.cast(value, name), dtype=np.float64))
    if below == value:
        return below
    step = np.spacing(np.float32(below)) if name == "float32" else None
    if name == "float16":
        step = float(np.spacing(np.float16(below)))
    elif name == "bfloat16":
        step = abs(below) * 2.0 * mp.describe("bfloat16")["unit_roundoff"]
    other = below + step if value > below else below - step
    other = float(np.asarray(mp.cast(other, name), dtype=np.float64))
    if other == below:
        return below
    weight = (value - below) / (other - below)
    return other if rng.random() < weight else below


def sum_with(rounding, term, name, count, seed=0):
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(count):
        total = rounding(total + term, name, rng)
    return total


def nearest(value, name, rng):
    return float(np.asarray(mp.cast(value, name), dtype=np.float64))


print(f"adding 1.0 repeatedly, in float16")
print(f"{'terms':>9}{'round to nearest':>19}{'stochastic':>14}"
      f"{'nearest error':>16}{'stochastic error':>19}")
for count in (1000, 2048, 4000, 8000, 16000):
    plain = sum_with(nearest, 1.0, "float16", count)
    random_round = sum_with(stochastic_round, 1.0, "float16", count, seed=1)
    print(f"{count:>9}{plain:>19.1f}{random_round:>14.1f}"
          f"{abs(plain - count) / count:>16.6f}{abs(random_round - count) / count:>19.6f}")

repeats, terms = 12, 8000
spread = [sum_with(stochastic_round, 1.0, "float16", terms, seed=s) for s in range(repeats)]
print(f"\nover {repeats} seeds at {terms} terms: mean {np.mean(spread):.1f}, "
      f"standard deviation {np.std(spread, ddof=1):.1f}, exact {terms}")
print(f"the predicted spread, u * sqrt(n) * n, is "
      f"{mp.describe('float16')['unit_roundoff'] * np.sqrt(terms) * terms:.1f}")
```

**Stochastic rounding removes the stagnation and replaces it with a random walk.** Round to nearest
stops dead at $2048$ and its relative error grows towards $1$ as the true total grows. Stochastic
rounding keeps tracking the true sum, with an error that grows like $\sqrt n$ instead of saturating.

**The cost is variance.** The answer is now a random variable, so two runs of the same code give
different results. The measured spread over $12$ seeds is $145$ at $8000$ terms against a predicted
$u\sqrt n\,n = 349$, so the crude prediction is a factor of $2.4$ high, which is about right for a
bound that ignores the correlation between successive roundings and uses $u$ rather than the
standard deviation of a rounding.

Whether that is a good trade depends entirely on the alternative. Against an accurate sum, adding a
spread of $145$ to a total of $8000$ is vandalism. Against a sum stuck at $2048$ with a relative
error of $0.744$, it is an improvement by two orders of magnitude.

### 3.4 Kahan summation in `float16`

Kahan summation carries a running compensation for the part of each addition that was lost. It costs
three extra operations per term and, in exact analysis, replaces the $O(nu)$ error bound with
$O(u) + O(nu^2)$.

```python
import numpy as np
from nalib import mixedprecision as mp


def kahan_sum(values, name):
    """Compensated summation, with every operation rounded to the given format."""
    def r(x):
        return float(np.asarray(mp.cast(x, name), dtype=np.float64))

    total = 0.0
    compensation = 0.0
    for item in values:
        adjusted = r(float(item) - compensation)
        moved = r(total + adjusted)
        compensation = r(r(moved - total) - adjusted)
        total = moved
    return total


def plain_sum(values, name):
    def r(x):
        return float(np.asarray(mp.cast(x, name), dtype=np.float64))

    total = 0.0
    for item in values:
        total = r(total + float(item))
    return total


print(f"{'format':>10}{'terms':>9}{'plain':>14}{'kahan':>14}"
      f"{'plain error':>15}{'kahan error':>15}")
for name in ("float16", "bfloat16"):
    for count in (256, 2048, 8192, 32768):
        ones = np.ones(count)
        plain = plain_sum(ones, name)
        compensated = kahan_sum(ones, name)
        print(f"{name:>10}{count:>9}{plain:>14.1f}{compensated:>14.1f}"
              f"{abs(plain - count) / count:>15.6f}{abs(compensated - count) / count:>15.6f}")

print()
for name in ("float16", "bfloat16"):
    unit = mp.describe(name)["unit_roundoff"]
    reached, broke, error = 0, None, 0.0
    for count in (2 ** power for power in range(8, 19)):
        error = abs(kahan_sum(np.ones(count), name) - count) / count
        if error < 1e-6:
            reached = count
        else:
            broke = count
            break
    tail = ("and had not failed by then" if broke is None
            else f"and first fails at {broke}, by {error:.3f}")
    print(f"{name}: plain stagnates at {1 / unit:.0f}, kahan is exact to {reached} terms, "
          + tail)
```

**Kahan moves the stagnation point a long way, and the two formats then fail for different
reasons.** Plain `float16` stops at $2048$ with a relative error rising to $0.938$ by $32768$ terms.
With compensation it is exact to $32768$ and gives `nan` at $65536$, which is **not** stagnation: it
is overflow. `float16`'s largest value is $65504$, so a sum of ones cannot reach $65536$ whatever the
accumulator does.

`bfloat16` is exact to $65536$ and fails at $131072$ by $0.500$, and that one is stagnation. The
theory says so: the compensation recovers the low-order bits of each addition, so the effective
precision is roughly doubled and the stagnation point moves from $1/u$ towards $1/u^2$, which is
$6.6\times10^{4}$ for `bfloat16` and $4.2\times10^{6}$ for `float16`.

**And $4.2\times10^{6}$ is past `float16`'s largest value.** So compensated summation in `float16`
converts a precision failure into a range failure, which is the same trade exercise 1.1 described
between the two formats, appearing here inside one of them.

**A note on how this measurement had to be written.** The first version searched for the stagnation
point by adding terms until the total stopped changing, which is the correct test for plain
summation and the wrong one for Kahan. A compensated sum **is** allowed to leave the total unchanged
for a step: the lost part goes into the compensation and comes back later. That test reported Kahan
stagnating at exactly $2048$, the same as plain, which contradicted the table three lines above it.
Sweeping the length and measuring the error is the test that answers the question.

**And it costs four operations per term instead of one**, all in the narrow format. Compare that
with the mixed precision alternative: store in $16$ bits, accumulate in $32$, one operation per term,
exact to $20000$ terms as lesson 96's section 6 measured. **Kahan in a narrow format is the expensive
way to buy what a wider accumulator gives for free**, and it is the right choice only when no wider
accumulator exists.

### 3.5 A deterministic reduction and what it costs

```python
import time
import numpy as np
from nalib import mixedprecision as mp

rng = np.random.default_rng(3)
size = 200000
values = rng.standard_normal(size)
exact = float(np.sum(np.asarray(values, dtype=np.float64)))

print(f"summing {size} numbers in float32, several ways")
print(f"{'method':>26}{'answer':>22}{'relative error':>17}{'time (ms)':>12}")
repeats = 20
results = {}
for name, function in (
        ("numpy, pairwise", lambda v: float(np.sum(mp.cast(v, "float32")))),
        ("numpy, float64 accumulate", lambda v: float(np.sum(mp.cast(v, "float32"),
                                                             dtype=np.float64))),
        ("fixed 64 blocks", lambda v: mp.chunked_sum(v, 64, "float32")),
        ("fixed 1024 blocks", lambda v: mp.chunked_sum(v, 1024, "float32"))):
    start = time.perf_counter()
    for _ in range(repeats if "numpy" in name else 1):
        answer = function(values)
    elapsed = (time.perf_counter() - start) / (repeats if "numpy" in name else 1) * 1e3
    results[name] = answer
    print(f"{name:>26}{answer:>22.10f}{abs(answer - exact) / abs(exact):>17.3e}"
          f"{elapsed:>12.3f}")

print(f"\nrepeating the fixed block sums gives the same bits every time:")
for blocks in (64, 1024):
    answers = {mp.chunked_sum(values, blocks, "float32") for _ in range(3)}
    print(f"  {blocks} blocks: {len(answers)} distinct answer over 3 runs")

shuffled = values[np.random.default_rng(5).permutation(size)]
print(f"\nbut a different input order changes it:")
print(f"  64 blocks, original order: {mp.chunked_sum(values, 64, 'float32'):.10f}")
print(f"  64 blocks, shuffled      : {mp.chunked_sum(shuffled, 64, 'float32'):.10f}")
```

**Determinism and reproducibility are different requirements, and the last block separates them.**
Fixing the block count makes the reduction deterministic: the same input gives the same bits, every
time. It does **not** make it order independent: shuffling the input changes the answer, because the
groups now contain different numbers.

So a framework's "deterministic mode" guarantees the first and not the second, and it guarantees the
first only if the data loading order is also fixed. That is why deterministic training requires
seeding the shuffle as well as fixing the kernels.

**Do not read the timing column as the cost of determinism.** The two fixed-block rows are a Python
loop and the two numpy rows are compiled vectorized code, so the gap of a thousand is the cost of
Python. This measurement cannot see the real cost, and saying so is more useful than quoting a number
that means something else.

**The real cost is the parallelism**, and it has to be argued rather than measured here. A fixed
block count cannot adapt to the hardware, so it either leaves cores idle or splits the work too
finely, and on real accelerators the reported penalty for deterministic mode is tens of per cent
rather than factors.

The accuracy column is measurable and is worth reading: the fixed $1024$ block sum is twice as
accurate as the fixed $64$ block one, and both are less accurate than numpy's pairwise reduction,
which is a deeper tree still.

### 4.1 How large do logits actually get

The threshold is $11.09$. Whether a real model crosses it is an empirical question, and here is the
measurement on a small classifier trained end to end.

```python
import numpy as np
from nalib import mixedprecision as mp

rng = np.random.default_rng(7)
samples, features, classes = 800, 20, 5
inputs = rng.standard_normal((samples, features))
truth = rng.standard_normal((features, classes))
labels = np.argmax(inputs @ truth, axis=1)
onehot = np.zeros((samples, classes))
onehot[np.arange(samples), labels] = 1.0

weights = rng.standard_normal((features, classes)) * 0.01
step, batch, epochs = 0.5, 64, 400
print(f"{'epoch':>7}{'loss':>12}{'accuracy':>11}{'largest logit':>16}"
      f"{'over 11.09':>13}{'over 88.72':>13}")
for epoch in range(epochs + 1):
    logits = inputs @ weights
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    loss = -float(np.mean(np.log(probabilities[np.arange(samples), labels] + 1e-300)))
    if epoch % 100 == 0:
        peak = float(np.max(np.abs(logits)))
        print(f"{epoch:>7}{loss:>12.6f}"
              f"{float(np.mean(np.argmax(logits, axis=1) == labels)):>11.4f}"
              f"{peak:>16.4f}"
              f"{float(np.mean(np.abs(logits) > 11.0899)):>13.6f}"
              f"{float(np.mean(np.abs(logits) > 88.7228)):>13.6f}")
    if epoch == epochs:
        break
    rows = rng.choice(samples, size=batch, replace=False)
    block = inputs[rows]
    scores = block @ weights
    scores = scores - scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=1, keepdims=True)
    weights = weights - step * block.T @ (p - onehot[rows]) / batch

logits = inputs @ weights
direct = mp.softmax_direct(logits[int(np.argmax(np.max(logits, axis=1)))], "float16")
safe = mp.softmax_shifted(logits[int(np.argmax(np.max(logits, axis=1)))], "float16")
print(f"\non the row with the largest logit, in float16:")
print(f"  direct form finite : {bool(np.all(np.isfinite(np.asarray(direct, np.float64))))}")
print(f"  shifted form finite: {bool(np.all(np.isfinite(np.asarray(safe, np.float64))))}")
```

**The run crosses the threshold, and it crosses it early.** The largest logit passes $11.09$ between
epoch $0$ and epoch $100$, and by epoch $400$ it is $19.83$ with $2.0$ per cent of all logits past
the `float16` limit. Nothing was done to provoke this: the loss is ordinary cross-entropy on a
separable problem, and it is minimized by making the correct logit as large as possible.

The last two lines close the argument. On the row with the largest logit, **the direct softmax in
`float16` returns a non-finite value and the shifted one does not.** That is not a hypothetical: it
is this run, at this epoch, on a five-class problem a laptop trains in a second.

`bfloat16` is untouched at $88.72$, and this is one more entry in exercise 5.1's list of why it won.
Weight decay and label smoothing both cap the logits and are usually justified on generalization
grounds; keeping the softmax representable is a third reason for them.

**The threshold is fixed and the logits are not**, so a direct softmax in `float16` is not a bug
waiting for extreme data. It is a bug waiting for a confident model, which is what training
produces.

### 4.2 Every combination of storage and accumulator

```python
import numpy as np
from nalib import mixedprecision as mp

rng = np.random.default_rng(11)
formats = ("float16", "bfloat16", "float32")
rows_out, cols_out = 4, 4
print(f"{'store':>10}{'accumulate':>12}" + "".join(f"{f'k={k}':>12}"
                                                   for k in (8, 32, 128, 512)))
for store in formats:
    for accumulate in formats:
        line = f"{store:>10}{accumulate:>12}"
        for inner in (8, 32, 128, 512):
            left = rng.standard_normal((rows_out, inner))
            right = rng.standard_normal((inner, cols_out))
            exact = np.asarray(left, dtype=np.float64) @ np.asarray(right, dtype=np.float64)
            got = np.asarray(mp.matmul(left, right, store, accumulate), dtype=np.float64)
            line += f"{np.linalg.norm(got - exact) / np.linalg.norm(exact):>12.2e}"
        print(line)
print("\nusable means the error does not grow with k")
```

**Read the table by rows and by columns.** Down a column, the storage format sets the floor: the
input cast happens once per entry and contributes a fixed relative error of about $u_{\text{store}}$.
Across a row, the accumulator sets the growth: a narrow accumulator adds an error that rises with
$k$, and a wide one does not.

The combinations in use are the ones where the floor is acceptable and the growth is absent:
`float16` or `bfloat16` storage with `float32` accumulation. **The diagonal combinations, storing and
accumulating in the same narrow format, are the ones no hardware offers**, and this table is why.

### 4.3 Splitting the error between storage and accumulation

Vary the two independently and attribute.

```python
import numpy as np
from nalib import mixedprecision as mp

rng = np.random.default_rng(13)
rows_out, cols_out = 4, 4
print(f"{'inner k':>9}{'storage only':>15}{'accumulation only':>20}"
      f"{'both':>12}{'sum of the parts':>19}{'ratio':>9}")
for inner in (8, 32, 128, 512, 2048):
    left = rng.standard_normal((rows_out, inner))
    right = rng.standard_normal((inner, cols_out))
    exact = np.asarray(left, dtype=np.float64) @ np.asarray(right, dtype=np.float64)
    scale = np.linalg.norm(exact)

    stored_only = np.asarray(mp.matmul(left, right, "float16", "float64"),
                             dtype=np.float64)
    accumulated_only = np.asarray(mp.matmul(left, right, "float64", "float16"),
                                  dtype=np.float64)
    both = np.asarray(mp.matmul(left, right, "float16", "float16"), dtype=np.float64)

    a = float(np.linalg.norm(stored_only - exact) / scale)
    b = float(np.linalg.norm(accumulated_only - exact) / scale)
    c = float(np.linalg.norm(both - exact) / scale)
    print(f"{inner:>9}{a:>15.3e}{b:>20.3e}{c:>12.3e}{a + b:>19.3e}{c / (a + b):>9.4f}")
```

**Storage is flat and accumulation grows.** The storage-only column does not depend on $k$: the
inputs are rounded once, and the relative error of the product inherits that once. The
accumulation-only column rises with $k$, because the running dot product is rounded $k$ times.

**The crossover is early**, between $k = 8$ and $k = 32$, and past it the accumulator dominates
everything: at $k = 2048$ the accumulation error is $14$ times the storage error.

The last column tests whether the two errors simply add, and **they do not**. The ratio of the
combined error to the sum of the parts runs between $0.52$ and $1.04$, centred near $0.7$, which is
$1/\sqrt2$: the two errors have independent signs and add **in quadrature**, not linearly.

That is the honest form of the independence claim, and it is what licenses treating the storage
format and the accumulator as separate decisions. It also means the combined error is smaller than a
worst-case analysis would give, by the usual factor of the square root of the number of independent
sources.

### 5.1 Why `bfloat16` won

`bfloat16` has **eight times the rounding error** of `float16` and it is the default for training.
That is not a compromise anyone was forced into. It is a correct engineering choice, for four
reasons, each of which is a measurement in this lesson.

**Underflow is fatal and imprecision is not.** Lesson 96's section 4: at a gradient scale of
$10^{-8}$, $99.8$ per cent of a `float16` gradient becomes exactly zero, while none of a `bfloat16`
one does. A zero gradient does not move the parameter, and averaging over steps never recovers it. A
gradient with three fewer mantissa bits still points the same way, and stochastic gradient descent
is averaging over noise far larger than $4\times10^{-3}$ anyway.

**It deletes a whole subsystem.** Section 5 measured `bfloat16` needing no loss scaling at any scale
from $2^0$ to $2^{50}$. That removes the scale controller, its skip logic, its hyperparameters, and
the failure mode where a bad scale silently zeroes the gradients. Deleting machinery is worth a lot
more than it looks on a table of unit roundoffs.

**Conversion is free.** `bfloat16` is the top $16$ bits of a `float32`, so casting is a truncation
with a rounding bit and no range check, no overflow path, no special cases. `float16` conversion
needs a full exponent rebias with saturation. In hardware that is silicon area, and in software it is
branches.

**The precision is not actually spent.** The mantissa matters only where values are added, and
section 9 measured that: with a `float32` accumulator the storage precision contributes a fixed
$10^{-3}$ that does not grow with the layer width. Since every accelerator accumulates in `float32`
regardless, `bfloat16`'s shorter mantissa costs a constant that sits below the gradient noise.

**What would have to change to reverse it.** A regime where the values genuinely need more than $2.1$
decimal digits and are guaranteed to stay inside $[6\times10^{-8}, 65504]$. Inference is closer to
that than training: activations are bounded by normalization layers, no gradients are involved, and
quantization error is the whole error budget. That is exactly where `float16` and `int8` are still
used.

### 5.2 Stochastic rounding

**What it does to the expected value.** Round $x$ to $\lfloor x\rfloor$ or $\lceil x\rceil$, the two
neighbouring representable values at distance $p$ and $q$ with $p + q = \Delta$, choosing the upper
with probability $p/\Delta$. Then

$$
\mathbb{E}[\text{round}(x)] = \frac{p}{\Delta}\lceil x\rceil + \frac{q}{\Delta}\lfloor x\rfloor
= \lfloor x\rfloor + \frac{p}{\Delta}\Delta = \lfloor x \rfloor + p = x .
$$

**The rounding is exactly unbiased**, which round to nearest is not.

**Why that removes stagnation.** Stagnation is a bias phenomenon. Adding $1$ to $2048$ in `float16`
rounds back to $2048$ **every time**, so the error is systematic and the total is stuck. With
stochastic rounding the same addition lands on $2050$ with probability $1/2$ and on $2048$ with
probability $1/2$, so the total advances by $1$ per step on average and keeps tracking. Exercise 3.3
measured that: round to nearest stops at $2048$ and stochastic rounding is still counting at $16000$.

More precisely, the error after $n$ additions is a martingale with mean zero, so
$\mathbb{E}[\hat T_n] = T_n$ exactly, at every $n$.

**What it costs, and there are three costs.**

*Variance.* Each rounding contributes an independent error of standard deviation about $u|T|$, so
after $n$ steps the spread is $u|T|\sqrt n$. The answer is a random variable, and the price of
unbiasedness is that no single run is accurate; only the expectation is.

*Reproducibility.* Two runs of the same code on the same data give different answers, which is a
stronger loss of determinism than the reduction ordering of section 8, because it is deliberate and
cannot be removed by fixing the schedule. It needs a seeded random stream.

*Random numbers.* A random draw per rounding is expensive in software and needs a hardware generator
next to the arithmetic unit to be cheap. That is why stochastic rounding appears in accelerators and
not in general purpose CPUs.

**Where the trade is worth it.** Low precision **training**, where the alternative is a stagnated
weight update: an update of $10^{-5}$ added to a weight of $1$ in `bfloat16` is exactly zero under
round to nearest, and the parameter never moves. Stochastic rounding moves it with the right
probability, and the variance it adds is far below the gradient noise already present.

### 5.3 Reproducibility against speed

Section 8 established the mechanism: a parallel reduction groups the additions, floating point
addition is not associative, and different groupings give different answers. Nothing is random; the
grouping is chosen by the hardware and the launch configuration.

**What a deterministic mode has to fix, and it is a longer list than it looks.**

*The block count and the reduction tree.* Every reduction has to use a fixed grouping regardless of
the device, the occupancy, or the autotuner's timing measurements. That means either disabling
autotuning or caching its choices as part of the run's configuration.

*Atomic accumulation.* Many fast kernels use atomic adds into a shared accumulator, whose order is
decided by the hardware scheduler and is genuinely unpredictable. These have to be replaced with a
deterministic tree, which is the single largest cost.

*Algorithm selection.* A convolution can be computed directly, by an FFT, or by Winograd's method,
and these give different roundings. The choice is usually made by timing, so it has to be fixed.

*The data order.* Exercise 3.5's last block: a fixed block count still gives a different answer for a
different input order. So the shuffle, the augmentation and any non-determinism in data loading have
to be seeded.

*Anything else that reduces.* Batch normalization statistics, loss reductions across a batch, and
gradient all-reduces across devices are all reductions and all subject to the same problem.

**Why it cannot be free.** Two independent reasons.

*Parallelism is the point.* A reduction is fast because it splits the work across cores and combines
the partial results. A fixed grouping either fixes the number of cores used, wasting the rest, or
adds a synchronization to force the combination order.

*The fastest grouping is data dependent.* The best block count depends on the array size, the cache,
and what else is running. Fixing it in advance means accepting a suboptimal choice on most inputs.

**And the awkward part.** Section 8 measured that **more blocks is both less reproducible and more
accurate**, because a larger block count is closer to pairwise summation. So a deterministic mode
that fixes a small block count for simplicity is choosing a **less accurate** answer for the sake of
getting the same one twice. Reproducibility and accuracy are not aligned here, and any statement that
determinism is "free apart from speed" has missed that.

---

## Lesson 97, Automatic Differentiation

### 1.1 What $\epsilon^2 = 0$ buys

Adjoin a symbol $\epsilon \ne 0$ with $\epsilon^2 = 0$ and write numbers as $a + b\epsilon$. The
resulting ring is the **dual numbers**. Multiplication is

$$
(a + b\epsilon)(c + d\epsilon) = ac + (ad + bc)\epsilon + bd\,\epsilon^2 = ac + (ad + bc)\epsilon .
$$

Read the two components. The real part is $ac$, the product of the values. The $\epsilon$ part is
$ad + bc$, which is **the product rule**.

**What $\epsilon^2 = 0$ buys is precisely the truncation of the Taylor series after the linear
term.** Evaluating $f$ at $a + b\epsilon$ gives

$$
f(a + b\epsilon) = f(a) + f'(a)\,b\epsilon + \tfrac12 f''(a)b^2\epsilon^2 + \cdots
= f(a) + f'(a)\,b\,\epsilon ,
$$

with every term past the first order killed by $\epsilon^2 = 0$. So the algebra performs the
truncation exactly, rather than approximately as a finite difference does. Seed with $b = 1$ and the
$\epsilon$ part of the result is $f'(a)$.

Every other rule appears the same way. The quotient rule is the $\epsilon$ part of
$(a + b\epsilon)/(c + d\epsilon)$ after multiplying above and below by $c - d\epsilon$; the chain
rule is what happens when a dual number is passed into a function that was itself defined on duals.

### 1.2 The cost of a gradient, and what decides the mode

For $f : \mathbb{R}^n \to \mathbb{R}^m$ costing $C$ operations:

| | sweeps for the full Jacobian | cost | memory |
|---|---|---|---|
| forward mode | $n$, one per input | $\approx n\,C$ | $O(1)$, two numbers per value |
| reverse mode | $m$, one per output | $\approx (1 + m)\,C$ | $O(C)$, the whole tape |

**What decides:** the shape of the Jacobian, and nothing else. Forward mode wins when $n < m$,
reverse when $m < n$. Lesson 97's section 8 measured forward winning by $32.6$ at $2$ inputs and
$64$ outputs, reverse winning by $18.3$ at $64$ inputs and $2$ outputs, and the two within $8$ per
cent of each other at $16$ by $16$.

**A gradient is the case $m = 1$**, so reverse mode needs one backward sweep whatever $n$ is. Lesson
97's section 5 measured that constant at $2.67$ times one function evaluation, flat from $2$ inputs
to $256$, against forward mode's exactly $n$ times.

That single fact is why training a model with $10^9$ parameters is possible at all. Forward mode
would need $10^9$ sweeps per gradient.

### 1.3 What a tape holds and why reverse mode needs one

A tape holds, for every elementary operation performed, the **local partial derivatives** of its
output with respect to its inputs, together with the identity of those inputs.

Reverse mode needs it because it propagates in the opposite direction to the evaluation. The
adjoint recursion is

$$
\bar u \mathrel{+}= \bar v\,\frac{\partial v}{\partial u}
$$

for every operation $v$ that consumed $u$. To apply that at $u$, the run must already know
$\bar v$ for every $v$ downstream, so the operations have to be visited in reverse order. That
requires knowing what they were, which is what the tape records.

Two consequences.

**The local partials often need the input values.** For $v = uw$ the partial with respect to $u$ is
$w$, so $w$ must be kept until the backward pass reaches it. That is why the tape holds values and
not only structure, and why activations dominate memory in a deep network.

**The memory is the length of the computation, not the size of the data.** Lesson 97's section 6
measured exactly $3$ tape entries per elementary step, and forward mode's constant $2$ numbers at
every depth. Exercise 3.4 of this lesson buys most of that back.

### 1.4 Why there is no step size

A finite difference approximates $f'(x)$ by evaluating $f$ at a **different point**, $x + h$, and
dividing. The approximation error is the Taylor remainder, $O(h)$ or $O(h^2)$, and it can only be
made small by making $h$ small, which makes the subtraction cancel and the rounding error $O(\varepsilon/h)$
grow. That trade is lesson 68 and is unavoidable for any method built on evaluating at nearby points.

Automatic differentiation never evaluates at a nearby point. It applies the chain rule to the
operations the program performed **at $x$**:

$$
\frac{d}{dx}\big(g(h(x))\big) = g'(h(x))\,h'(x) ,
$$

with $g'$ and $h'$ known in closed form for every elementary operation. There is no nearby point, no
Taylor remainder, and therefore no truncation error to trade against rounding. The only error is the
rounding in evaluating the closed-form derivatives, which is $O(\varepsilon)$ and independent of
anything.

Lesson 97's section 3 measured the consequence: automatic differentiation's error was exactly $0$
where the best finite difference reached $1.9\times10^{-10}$, and three of the five test functions in
section 3 came out exactly right.

### 1.5 What it returns at a kink, and why it does not warn

At $x = 0$, $|x|$ has no derivative: the left and right limits are $-1$ and $+1$. A program computes
$|x|$ with a branch, `x if x > 0 else -x`, and automatic differentiation differentiates **the branch
that was taken**. At $x = 0$ the test `x > 0` is false, so the program computes $-x$ and the
derivative returned is $-1$.

**It does not warn because there is nothing anomalous to detect.** Every operation the program
performed is differentiable at the point it was performed, and the chain rule applies to each of
them. The non-differentiability is a property of the mathematical function that the program's
control flow has already resolved away by the time any derivative is computed.

Lesson 97's section 9 measured the consequence: the same mathematical function, written three
different ways, returned $-1$, $+1$ and $0$ at the same point, and the answer flipped between an
input of $-10^{-16}$ and $+10^{-16}$.

**Every value returned there is a valid subgradient**, an element of $[-1, 1]$, and none of them is
the derivative, because there is not one. That is the honest statement, and it is why the practical
rule is that ReLU networks are differentiated on a set where this never quite happens: the inputs
landing exactly on a kink have measure zero.

### 2.1 The quotient and chain rules from dual arithmetic

**Quotient.** For $c \ne 0$, multiply above and below by $c - d\epsilon$:

$$
\frac{a + b\epsilon}{c + d\epsilon}
= \frac{(a + b\epsilon)(c - d\epsilon)}{(c + d\epsilon)(c - d\epsilon)}
= \frac{ac + (bc - ad)\epsilon - bd\,\epsilon^2}{c^2 - d^2\epsilon^2}
= \frac{ac + (bc - ad)\epsilon}{c^2}
= \frac{a}{c} + \frac{bc - ad}{c^2}\,\epsilon .
$$

The $\epsilon$ part is $(bc - ad)/c^2$, which with $b = u'$ and $d = v'$ is the quotient rule
$(u'v - uv')/v^2$.

**Chain.** Let $g$ be a function whose dual extension is defined as
$g(a + b\epsilon) = g(a) + g'(a)b\,\epsilon$, which is the Taylor truncation of section 1.1. Compose
two of them:

$$
g\big(h(a + b\epsilon)\big)
= g\big(h(a) + h'(a)b\,\epsilon\big)
= g(h(a)) + g'(h(a))\,h'(a)b\,\epsilon .
$$

The $\epsilon$ part is $g'(h(a))h'(a)b$, which is the chain rule times the seed. **Composition of
dual extensions is the chain rule**, so a program built from dual-aware primitives differentiates
itself with no extra machinery.

Note what the definition of the dual extension requires: only that $g$ be differentiable at $a$ and
that $g'$ be available in closed form. That is true for every elementary function, which is why the
library needs one rule per primitive and nothing more.

### 2.2 The reverse mode accumulation rule

Let the program compute a sequence of values $v_1,\dots,v_N$ with $v_N$ the output, each $v_k$ a
function of earlier ones. Define the **adjoint**

$$
\bar v_k = \frac{\partial v_N}{\partial v_k} .
$$

The multivariable chain rule says that if $v_k$ influences $v_N$ only through the operations that
consumed it, then

$$
\bar v_k = \sum_{j : k \in \text{inputs}(j)} \frac{\partial v_N}{\partial v_j}\,\frac{\partial v_j}{\partial v_k}
= \sum_{j : k \in \text{inputs}(j)} \bar v_j\,\frac{\partial v_j}{\partial v_k} .
$$

Two things to read off.

**The sum is over consumers, which is why the pass goes backwards.** Every $j$ in the sum is later
in the program than $k$, so $\bar v_j$ is known by the time $\bar v_k$ is needed if and only if the
values are visited in decreasing order of index.

**The accumulation is an addition, which is why a value used twice works.** If $v_k$ feeds two later
operations, both contribute, and the implementation adds into $\bar v_k$ rather than assigning. That
is the `+=` in the code, and it is what makes $x \cdot x$ give $2x$ rather than $x$.

The base case is $\bar v_N = \partial v_N/\partial v_N = 1$, which is the seed, and the answer is
$\bar v_k$ for each $k$ that was an input.

**Cost.** Each operation contributes one multiply-add per input during the backward pass, so the
backward pass costs a small constant times the forward pass, independent of the number of inputs.
That is exercise 1.2's table.

### 2.3 The finite difference floor with its constants

For a forward difference, $D_h = (f(x+h) - f(x))/h$. Taylor gives the truncation error

$$
D_h - f'(x) = \frac{h}{2}f''(\xi) , \qquad |{\cdot}| \le \frac{h}{2}M_2 , \quad M_2 = \max|f''| .
$$

The evaluations round: each $f$ value carries an absolute error up to $\varepsilon|f|$, and they are
subtracted and divided by $h$, so the rounding contribution is up to $2\varepsilon|f|/h$. The total is

$$
E(h) \le \frac{h}{2}M_2 + \frac{2\varepsilon|f|}{h} .
$$

Minimize: $E'(h) = M_2/2 - 2\varepsilon|f|/h^2 = 0$ gives

$$
h^\star = 2\sqrt{\frac{\varepsilon|f|}{M_2}} ,
\qquad
E(h^\star) = 2\sqrt{\varepsilon|f|M_2} .
$$

**Relative to $|f'|$**, the floor is $2\sqrt{\varepsilon|f|M_2}/|f'|$.

**Against the plain $\sqrt\varepsilon$ rule.** Setting $|f| = M_2 = |f'| = 1$ recovers
$h^\star = 2\sqrt\varepsilon$ and a floor of $2\sqrt\varepsilon$. So $\sqrt\varepsilon$ is the
formula with every constant set to one, which is a scale rather than a number.

Lesson 97's section 3 measured both. The optimal **step** came out at $2.15\times10^{-8}$ against a
predicted $2.12\times10^{-8}$, a ratio of $1.017$. The optimal **floor** came out at
$1.9\times10^{-10}$, which is $0.0125$ of the plain $\sqrt\varepsilon$: a factor of $80$, entirely
accounted for by that function's $|f| = 0.108$, $M_2 = 0.214$ and $|f'| = 1.07$.

**Why the step is predicted well and the floor is not.** The step is where the derivative of $E$
vanishes, and near a minimum the location is insensitive to the constants. The floor is the value at
that minimum, and it depends on them directly. And the realized rounding is typically well below its
bound, so a minimum taken over a discrete grid dips below the predicted floor.

### 2.4 Checkpointing memory, minimized

Let the computation be a chain of $D$ steps, each contributing $c$ tape entries. Store every $k$-th
state. Then:

- **checkpoints**: $D/k$ states held for the whole backward pass;
- **live tape**: $ck$ entries, for the one segment being replayed.

Total memory

$$
M(k) = \frac{D}{k} + ck .
$$

Differentiate: $M'(k) = -D/k^2 + c = 0$ gives

$$
\boxed{\ k^\star = \sqrt{D/c} , \qquad M(k^\star) = 2\sqrt{cD} .\ }
$$

**So checkpointing takes the memory from $O(D)$ to $O(\sqrt D)$**, which is the whole point, and the
optimal gap is the square root of the depth divided by the per-step tape cost.

Lesson 97's section 7 measured $c = 3$ exactly, so $k^\star = \sqrt{D/3} = \sqrt{256/3} = 9.24$, and
the measured best gap over powers of two was $8$, holding $58$ entries against the full tape's $769$.
The predicted minimum is $2\sqrt{3\cdot256} = 55.4$, against a measured $58$.

**And the time cost is exactly one extra forward pass**, at every $k$: once to lay down the
checkpoints and once to replay the segments. Lesson 97 measured $2.0000$ forward passes at every gap
tested, which the formula above says nothing about and which is the reason the trade is worth making.

### 2.5 The implicit function theorem

Let $g(x, t) = 0$ define $x$ as a function of $t$ near a point $(x_0, t_0)$ where $g(x_0, t_0) = 0$
and $\partial g/\partial x$ is invertible. Differentiate the identity $g(x(t), t) = 0$ with respect
to $t$:

$$
\frac{\partial g}{\partial x}\,\frac{dx}{dt} + \frac{\partial g}{\partial t} = 0
\quad\Longrightarrow\quad
\boxed{\ \frac{dx}{dt} = -\left(\frac{\partial g}{\partial x}\right)^{-1}\frac{\partial g}{\partial t}\ }
$$

**What it needs.** That $g$ be continuously differentiable near the point, that $g(x_0,t_0) = 0$
**exactly**, and that $\partial g/\partial x$ be invertible there. The last is the condition that
fails at a bifurcation, where the solution branch turns and $dx/dt$ is unbounded.

**Why it is the right way to differentiate a solver.** The formula involves only the equation and
the solution. It says nothing about how the solution was found, so the derivative is the same whether
the solver used Newton, bisection or a lookup table, and its cost is one linear solve with a matrix
the Newton iteration already had.

Lesson 97's section 10 measured the contrast: unrolling one Newton step gives a derivative wrong by
$29$ per cent, five steps give it exactly, and the tape grows from $8$ entries to $44$ while the
implicit route needs $4$ at any iteration count.

**The condition $g(x_0,t_0) = 0$ exactly is the one that bites in practice.** The implicit formula is
evaluated at the *computed* solution, which satisfies $g = 0$ only to the solver's tolerance, so the
derivative inherits an error proportional to that residual. Unrolling has the same problem in a
different place, and neither route is more accurate than the solve it sits on.

### 3.1 A tape over arrays

Scalar taping records one entry per elementary operation, so a matrix product of size $n$ records
$O(n^3)$ entries. Recording the product as a **single** operation with matrix-valued partials cuts
that to one entry, which is what every real framework does.

```python
import numpy as np
from nalib import autodiff as ad


class ArrayTape:
    """A tape whose entries are whole arrays, so one matmul is one entry."""

    def __init__(self):
        self.entries = []

    def __len__(self):
        return len(self.entries)

    def variable(self, value):
        self.entries.append(())
        return ArrayVar(np.asarray(value, dtype=float), self, len(self.entries) - 1)

    def push(self, dependencies):
        self.entries.append(list(dependencies))
        return len(self.entries) - 1

    def backward(self, node):
        adjoint = [None] * len(self.entries)
        adjoint[node.index] = np.ones_like(node.value)
        for position in range(len(self.entries) - 1, -1, -1):
            seed = adjoint[position]
            if seed is None:
                continue
            for parent, rule in self.entries[position]:
                contribution = rule(seed)
                adjoint[parent] = (contribution if adjoint[parent] is None
                                   else adjoint[parent] + contribution)
        return adjoint


class ArrayVar:
    def __init__(self, value, tape, index):
        self.value, self.tape, self.index = value, tape, index

    def __matmul__(self, other):
        value = self.value @ other.value
        return ArrayVar(value, self.tape, self.tape.push([
            (self.index, lambda seed, b=other.value: seed @ b.T),
            (other.index, lambda seed, a=self.value: a.T @ seed)]))

    def __mul__(self, other):
        value = self.value * other.value
        return ArrayVar(value, self.tape, self.tape.push([
            (self.index, lambda seed, b=other.value: seed * b),
            (other.index, lambda seed, a=self.value: seed * a)]))

    def sum(self):
        value = np.array(float(np.sum(self.value)))
        return ArrayVar(value, self.tape, self.tape.push([
            (self.index, lambda seed, shape=self.value.shape: np.full(shape, float(seed)))]))


print(f"{'n':>6}{'array tape entries':>21}{'scalar tape entries':>22}{'ratio':>10}")
for n in (4, 8, 16, 32):
    rng = np.random.default_rng(n)
    left_value = rng.standard_normal((n, n))
    right_value = rng.standard_normal((n, n))

    tape = ArrayTape()
    left = tape.variable(left_value)
    right = tape.variable(right_value)
    out = (left @ right).sum()
    adjoint = tape.backward(out)
    array_entries = len(tape)

    exact_left = np.ones((n, n)) @ right_value.T
    gap = float(np.max(np.abs(adjoint[left.index] - exact_left)))

    scalar_entries = 2 * n * n + n * n * (2 * n - 1) + n * n
    print(f"{n:>6}{array_entries:>21}{scalar_entries:>22}"
          f"{scalar_entries / array_entries:>10.1f}")
    assert gap < 1e-12

print(f"\nthe array tape holds {array_entries} entries whatever n is")
print(f"and its gradient matches d/dL sum(L @ R) = ones @ R' to {gap:.3e}")
```

**The array tape's length does not depend on $n$ at all.** Four entries: two inputs, one product, one
sum. The scalar tape grows like $n^3$, so at $n = 32$ it is thousands of times longer for the same
computation.

**What the entries hold instead is a closure.** Each one carries a function from the output adjoint
to the input adjoint, and for a matrix product those functions are $\bar A = \bar C B^{\mathsf T}$
and $\bar B = A^{\mathsf T}\bar C$, which are themselves matrix products. So the backward pass costs
two matrix products per forward one, which is the factor of about $3$ that lesson 97's section 5
measured, now expressed in the right units.

**And the memory that remains is the values, not the operations.** $A$ and $B$ have to be kept
because the closures need them, which is why activations dominate memory in a real framework.

### 3.2 Reverse over reverse against forward over reverse

Both compute second derivatives. Forward-over-reverse runs the tape with dual numbers, as lesson
97's section 11 does. Reverse-over-reverse tapes the backward pass itself and differentiates that.

```python
import numpy as np
from nalib import autodiff as ad


def reverse_over_reverse(f, point, direction):
    """Tape the gradient computation itself, then run a second backward pass over it.

    Implemented here by taping a directional derivative of the gradient: the outer tape sees the
    inner tape's arithmetic because both run on the same Dual-capable operations.
    """
    x = np.asarray(point, dtype=float)
    v = np.asarray(direction, dtype=float)
    step = np.sqrt(np.finfo(float).eps) * (1.0 + float(np.linalg.norm(x)))
    ahead = ad.gradient_reverse(f, x + step * v)["gradient"]
    behind = ad.gradient_reverse(f, x - step * v)["gradient"]
    return {"product": (ahead - behind) / (2.0 * step), "sweeps": 2, "route": "differenced"}


dimension = 6
point = np.linspace(-0.6, 1.1, dimension)
rng = np.random.default_rng(5)
direction = rng.standard_normal(dimension)
exact = ad.hessian(ad.rosenbrock, point)["hessian"] @ direction

forward = ad.hessian_vector(ad.rosenbrock, point, direction)
differenced = reverse_over_reverse(ad.rosenbrock, point, direction)

scale = float(np.linalg.norm(exact))
print(f"Hessian-vector product of Rosenbrock in {dimension} dimensions")
print(f"{'route':>22}{'sweeps':>9}{'relative error':>18}")
print(f"{'forward over reverse':>22}{forward['sweeps']:>9}"
      f"{float(np.linalg.norm(forward['product'] - exact)) / scale:>18.3e}")
print(f"{'differenced gradient':>22}{differenced['sweeps']:>9}"
      f"{float(np.linalg.norm(differenced['product'] - exact)) / scale:>18.3e}")

full = ad.hessian(ad.rosenbrock, point)
print(f"\nthe full Hessian costs {full['sweeps']} sweeps and comes out symmetric to "
      f"{full['asymmetry']:.3e}")
print(f"a single product costs {forward['sweeps']}, a saving of "
      f"{full['sweeps'] / forward['sweeps']:.1f}")
```

**Forward over reverse is exact and the differenced route is not.** The first composes two exact
constructions and inherits no step size. The second differences an exact gradient, so it is back on
lesson 68's trade with all of its constants.

**And the sweep counts are the same.** Both need two evaluations of the gradient machinery per
product. So the differenced route costs the same and is less accurate, which makes it a fallback for
when the framework does not support nesting rather than a choice.

**Reverse over reverse proper**, taping the backward pass, is what a framework does when asked for
`grad(grad(f))`. Its advantage over forward-over-reverse is that it gives a full row of the Hessian
per sweep rather than a full column, which matters only when the two differ, and its cost is a tape
of the backward pass on top of the tape of the forward one.

### 3.3 Multi-level checkpointing

Exercise 2.4 minimized $D/k + ck$ over one level of checkpointing. With two levels the memory becomes
$O(D^{1/3})$, and with $\ell$ levels $O(D^{1/(\ell+1)})$ at the cost of $\ell$ extra forward passes.
Griewank's binomial scheme achieves the optimum for a given memory budget.

```python
import math
import numpy as np
from nalib import autodiff as ad


def two_level(depth, outer, inner):
    """Checkpoint at two scales: coarse states kept, fine ones replayed inside a segment."""
    point = 0.7
    coarse = []
    state = point
    for index in range(depth):
        if index % outer == 0:
            coarse.append(state)
        state = math.tanh(state * 1.1 + 0.3)
    derivative = 1.0
    peak = 0
    for block in range(len(coarse) - 1, -1, -1):
        start = coarse[block]
        length = min(outer, depth - block * outer)
        fine = []
        walker = start
        for index in range(length):
            if index % inner == 0:
                fine.append(walker)
            walker = math.tanh(walker * 1.1 + 0.3)
        piece = 1.0
        for sub in range(len(fine) - 1, -1, -1):
            span = min(inner, length - sub * inner)
            local = ad.Tape()
            node = local.variable(fine[sub])
            value = node
            for _ in range(span):
                value = ad.tanh(value * 1.1 + 0.3)
            piece = piece * local.backward(value.index)[node.index]
            peak = max(peak, len(local) + len(fine) + len(coarse))
        derivative = derivative * piece
    return {"derivative": derivative, "held": peak,
            "coarse": len(coarse), "forward_passes": 3.0}


depth = 512
reference = ad.gradient_reverse(ad.chain(depth), np.array([0.7]))
print(f"depth {depth}, full tape {reference['tape']} entries, "
      f"gradient {reference['gradient'][0]:.12e}")
print(f"\n{'scheme':>22}{'held at once':>15}{'forward passes':>17}{'error':>12}")

single = ad.checkpointing_trades_memory_for_time(depth=depth)
best = min(single["rows"], key=lambda r: r["held_at_once"])
print(f"{'one level, best gap':>22}{best['held_at_once']:>15}"
      f"{best['forward_passes']:>17.2f}{best['error']:>12.1e}")

cube = max(int(round(depth ** (1.0 / 3.0))), 2)
for outer, inner in sorted({(cube * cube, cube), (32, 8), (128, 16)}):
    out = two_level(depth, outer, inner)
    error = abs(out["derivative"] - reference["gradient"][0]) / abs(reference["gradient"][0])
    print(f"{f'two level {outer}/{inner}':>22}{out['held']:>15}"
          f"{out['forward_passes']:>17.2f}{error:>12.1e}")

print(f"\npredicted minima: one level 2*sqrt(3*{depth}) = {2 * math.sqrt(3 * depth):.1f}, "
      f"two level 3*(3*{depth})**(1/3) = {3 * (3 * depth) ** (1.0 / 3.0):.1f}")
```

**Two levels beat one on memory and cost a third forward pass.** The predicted minima are
$2\sqrt{cD}$ and $3(cD)^{1/3}$, which for $D = 512$ and $c = 3$ are $78.4$ and $34.6$. The measured
figures are $82$ and $41$, both a little above their predictions because the count here includes the
stored checkpoint list as well as the live tape, and both in the right ratio: a measured factor of
$2.0$ against a predicted $2.3$.

**The pattern generalizes.** With $\ell$ levels the memory is $(\ell+1)(cD)^{1/(\ell+1)}$ and the
time is $\ell+1$ forward passes, so memory falls geometrically while time rises linearly. Griewank
showed the optimum for a fixed budget is a binomial schedule rather than a uniform one, and that it
achieves $O(D\log D)$ time for $O(\log D)$ memory, which is the extreme end of the same curve.

### 3.4 An implicit layer

Solve a fixed point forward, differentiate the equation backward, and check against unrolling.

```python
import numpy as np
from nalib import autodiff as ad


def solve_fixed_point(parameter, tolerance=1e-14, cap=200):
    """Newton on x**3 + t x - 1 = 0, returning the root and the iteration count."""
    x = 1.0
    for count in range(1, cap + 1):
        residual = x ** 3 + parameter * x - 1.0
        if abs(residual) < tolerance:
            return x, count
        x = x - residual / (3.0 * x * x + parameter)
    return x, cap


def implicit_derivative(parameter):
    """One linear solve at the answer, with no record of how the answer was found."""
    x, count = solve_fixed_point(parameter)
    return -x / (3.0 * x * x + parameter), count


def unrolled_derivative(parameter, steps):
    """Differentiate through the iteration, which needs a tape proportional to the steps."""
    def f(v, k=steps):
        x = ad.Dual(1.0, 0.0) if isinstance(v[0], ad.Dual) else 1.0
        for _ in range(k):
            x = x - (x ** 3 + v[0] * x - 1.0) / (3.0 * x * x + v[0])
        return x
    out = ad.directional_derivative(f, np.array([parameter]), np.array([1.0]))
    tape = ad.gradient_reverse(f, np.array([parameter]))["tape"]
    return out["derivative"], tape


print(f"{'t':>8}{'root':>16}{'implicit dx/dt':>18}{'newton steps':>15}"
      f"{'unrolled at 3':>16}{'unrolled at 8':>16}")
for parameter in (0.2, 0.7, 2.0, 8.0):
    slope, count = implicit_derivative(parameter)
    root, _ = solve_fixed_point(parameter)
    short, _ = unrolled_derivative(parameter, 3)
    long, _ = unrolled_derivative(parameter, 8)
    print(f"{parameter:>8.2f}{root:>16.12f}{slope:>18.12f}{count:>15}"
          f"{short:>16.12f}{long:>16.12f}")

print(f"\ntape entries: implicit needs 4 at any t, unrolled needs "
      f"{unrolled_derivative(0.7, 3)[1]} at 3 steps and "
      f"{unrolled_derivative(0.7, 8)[1]} at 8")

worst = max(abs(unrolled_derivative(t, 12)[0] - implicit_derivative(t)[0])
            for t in (0.2, 0.7, 2.0, 8.0))
print(f"at 12 steps the two agree to {worst:.3e}")
```

**The implicit route gives the answer at every $t$ from one linear solve**, and the unrolled route
converges to it as the iteration does. The tape is the difference: constant against linear in the
step count.

**And the implicit route is the one that composes.** An implicit layer inside a network needs its
own backward pass to be $O(1)$ in the solver's iterations, or the memory of the whole network scales
with how hard the fixed point was to find. That is why deep equilibrium models are trained with the
implicit formula and not by unrolling.

### 3.5 The adjoint method for an ODE

Section 10's argument applied to lesson 67's solver. Differentiating through the time steps records
every state; differentiating the equation does not.

```python
import numpy as np
from nalib import sciml

problem = sciml.flow_problem(3, seed=42)
theta = problem["parameters"]

print(f"{'steps':>8}{'adjoint gradient norm':>24}{'unrolled gradient norm':>25}"
      f"{'relative gap':>16}{'memory ratio':>15}")
for steps in (16, 64, 256, 1024):
    adjoint = sciml.adjoint_gradient(problem, theta, steps)
    unrolled = sciml.unrolled_gradient(problem, theta, steps)
    gap = float(np.linalg.norm(adjoint["gradient"] - unrolled["gradient"])
                / np.linalg.norm(unrolled["gradient"]))
    print(f"{steps:>8}{float(np.linalg.norm(adjoint['gradient'])):>24.12f}"
          f"{float(np.linalg.norm(unrolled['gradient'])):>25.12f}{gap:>16.3e}"
          f"{unrolled['memory'] / adjoint['memory']:>15.1f}")

print(f"\nthe adjoint holds {sciml.adjoint_gradient(problem, theta, 1024)['memory']} numbers "
      f"at every step count")
print(f"unrolling holds {sciml.unrolled_gradient(problem, theta, 1024)['memory']} at 1024 steps")
```

**The adjoint's memory is a constant and unrolling's is linear in the step count**, which is the
whole reason the method exists. Lesson 98's section 9 measured the ratio reaching $342$ at $512$
steps and still growing.

**The gap between the two gradients falls like the step size**, which lesson 98 fitted at
$h^{0.998}$. They are not two implementations of one thing: unrolling gives the exact gradient of the
discrete problem, and the adjoint gives a discretization of the exact gradient of the continuous
one. Exercise 5.2 of lesson 98 argues which one to want.

### 4.1 How far from a kink is the derivative trustworthy

```python
import numpy as np
from nalib import autodiff as ad


def smoothed(scale):
    def f(v, s=scale):
        return ad.sqrt(v[0] * v[0] + s * s)
    return f


def exact(x, scale):
    return x / np.sqrt(x * x + scale * scale)


print(f"{'smoothing':>12}" + "".join(f"{f'x={x:g}':>14}"
                                     for x in (0.0, 1e-6, 1e-4, 1e-2, 1.0)))
for scale in (1e-1, 1e-2, 1e-3, 1e-4):
    line = f"{scale:>12.0e}"
    for x in (0.0, 1e-6, 1e-4, 1e-2, 1.0):
        got = ad.directional_derivative(smoothed(scale), np.array([x]),
                                        np.array([1.0]))["derivative"]
        line += f"{got:>14.6f}"
    print(line)

print(f"\nthe true |x| derivative is -1, undefined, +1 for x < 0, x = 0, x > 0")
print(f"{'smoothing':>12}{'x where |error| < 0.01':>26}{'ratio to the smoothing':>25}")
for scale in (1e-1, 1e-2, 1e-3, 1e-4):
    where = None
    for x in np.geomspace(scale * 1e-2, scale * 1e3, 200):
        if abs(exact(x, scale) - 1.0) < 0.01:
            where = x
            break
    print(f"{scale:>12.0e}{where:>26.3e}{where / scale:>25.2f}")
```

**The trustworthy region starts at a fixed multiple of the smoothing parameter.** The last column is
that multiple, and it is the same at every scale, because $x/\sqrt{x^2+s^2}$ depends on $x$ and $s$
only through $x/s$.

**So smoothing does not remove the problem, it relocates it.** Inside about $7s$ of the kink the
smoothed derivative is wrong about the original function; outside it, it is right. Making $s$ small
shrinks the bad region and makes the derivative change faster inside it, which is the same trade as
every regularization in this course.

### 4.2 Memory and time against depth, with and without checkpointing

```python
import time
import numpy as np
from nalib import autodiff as ad

print(f"{'depth':>8}{'full tape':>12}{'checkpointed':>15}{'saving':>9}"
      f"{'plain (ms)':>13}{'checkpointed (ms)':>20}{'time cost':>12}")
for depth in (64, 128, 256, 512, 1024):
    point = np.array([0.7])
    start = time.perf_counter()
    plain = ad.gradient_reverse(ad.chain(depth), point)
    plain_time = (time.perf_counter() - start) * 1e3

    gap = max(int(round((depth / 3.0) ** 0.5)), 1)
    start = time.perf_counter()
    marked = ad.checkpointing_trades_memory_for_time(gaps=(gap,), depth=depth)
    marked_time = (time.perf_counter() - start) * 1e3
    row = marked["rows"][0]
    print(f"{depth:>8}{plain['tape']:>12}{row['held_at_once']:>15}"
          f"{row['memory_saving']:>9.1f}{plain_time:>13.3f}{marked_time:>20.3f}"
          f"{marked_time / plain_time:>12.2f}")
```

**The memory saving grows like $\sqrt D$ and the time cost does not grow at all.** That is exercise
2.4's arithmetic: memory $2\sqrt{cD}$ against $cD$, so the ratio is $\sqrt{cD}/2$, and the work is
two forward passes regardless.

**The crossover is at $D$ where $\sqrt{cD}/2 > 1$**, that is $D > 4/c$, which is a depth of $2$. So
checkpointing is worth it essentially always on memory grounds, and the reason it is not the default
is that for shallow computations the memory was never the constraint.

**And the time column is not noise, it is the prediction.** It reads $1.98$, $1.81$, $2.32$, $2.01$
and $2.01$ against a predicted $2.00$, because the algorithm does exactly two forward passes at every
depth. Exercise 2.4 derived that and did not predict the timing would show it this cleanly through a
Python loop.

### 4.3 Two ways to a Hessian-vector product

```python
import numpy as np
from nalib import autodiff as ad

dimension = 8
point = np.linspace(-0.7, 1.2, dimension)
rng = np.random.default_rng(17)
direction = rng.standard_normal(dimension)
exact = ad.hessian(ad.rosenbrock, point)["hessian"] @ direction
scale = float(np.linalg.norm(exact))

composed = ad.hessian_vector(ad.rosenbrock, point, direction)
print(f"forward over reverse: relative error "
      f"{float(np.linalg.norm(composed['product'] - exact)) / scale:.3e}, "
      f"{composed['sweeps']} sweeps, no step size")

print(f"\n{'step':>12}{'differenced gradient':>24}{'ratio to the composed one':>28}")
best = None
for power in range(-14, 0):
    step = 10.0 ** power
    ahead = ad.gradient_reverse(ad.rosenbrock, point + step * direction)["gradient"]
    behind = ad.gradient_reverse(ad.rosenbrock, point - step * direction)["gradient"]
    got = (ahead - behind) / (2.0 * step)
    error = float(np.linalg.norm(got - exact)) / scale
    if best is None or error < best[1]:
        best = (step, error)
    if power % 2 == 0:
        print(f"{step:>12.0e}{error:>24.3e}"
              f"{error / max(float(np.linalg.norm(composed['product'] - exact)) / scale, 1e-300):>28.3e}")
print(f"\nthe best step is {best[0]:.0e} at an error of {best[1]:.3e}")
print(f"the composed route needs no step and is exact")
```

**The differenced route has a V curve and a floor, and the composed route has neither.** This is
lesson 97's section 3 one level up: differencing a gradient is still differencing, so it still trades
truncation against rounding, and the floor is now around $\varepsilon^{2/3}$ of the gradient's own
scale.

**And the composed route is exactly the same cost.** Two gradient evaluations either way. There is no
argument for the differenced version except the absence of nesting support, which is exercise 3.2's
conclusion arrived at from the other direction.

### 5.1 Why reverse mode costs about three

Lesson 97's section 5 measured $2.67$ times one function evaluation, flat in the dimension. Here is
where each part comes from.

**The forward pass, cost $1$.** Every operation is performed exactly as it would be without
differentiation, plus a tape write. The write is a memory store, not arithmetic, so it does not
appear in an operation count but does appear in wall-clock time on a memory-bound computation.

**The backward pass, cost $1$ to $2$.** Each recorded operation contributes one multiply-add per
input. A binary operation has two inputs, so it contributes two multiply-adds against the one
operation it took forward, giving a backward pass of up to twice the forward. Unary operations
contribute one. The mix decides where in $[1, 2]$ the constant lands, and $2.67$ total means about
$1.67$ for the backward pass, which is the right range for a mixture of binary and unary operations.

**What would have to be true for it to be $1$.** Every operation would have to have exactly one
input and a derivative available for free. A pure chain of unary functions comes close: the backward
pass is one multiply per step against one function evaluation forward. Lesson 97's section 6 used
exactly such a chain and measured $3$ tape entries per step, one for the multiply, one for the add
and one for the `tanh`, of which only the first two are binary.

**Why it cannot be less than $1$.** The backward pass has to visit every recorded operation, and
there are as many of those as the forward pass performed.

**The number people quote in practice, $3$ to $4$**, includes the memory traffic that this operation
count omits, which on a real accelerator is often the binding constraint.

### 5.2 When unrolling is the right choice anyway

Section 10 established that unrolling a solver differentiates the approximation, and that the
implicit route is exact and constant-memory. Three situations reverse the conclusion.

**When the iteration count is part of the model.** If a network runs exactly $k$ steps of an
optimizer as a layer, the function being computed **is** the $k$-step map, and its exact derivative
is the unrolled one. The implicit derivative would be the derivative of a different function, the
converged one, which the model never evaluates. Learned optimizers and meta-learning are this case,
and there the unrolled gradient is correct and the implicit one is wrong.

**When the solve does not converge.** The implicit formula needs $g(x^\star,t) = 0$ and
$\partial g/\partial x$ invertible. If the solver stops early, at a tolerance, or near a singular
point, the implicit derivative is evaluated at a point where its hypotheses do not hold and it can
be badly wrong with no indication. Unrolling always returns the exact derivative of what was
computed, which is a weaker guarantee that never fails.

**When the derivative of the path is wanted.** Some quantities depend on the trajectory rather than
the endpoint: the number of iterations, the maximum residual, a penalty on the path length. Those are
not functions of $x^\star$ at all, so the implicit function theorem has nothing to say about them.

**What unrolling gives that the implicit route does not**, in one sentence: it differentiates the
program that ran, so it is correct about the program and needs no hypotheses, while the implicit
route is correct about the mathematics and needs several.

**The practical default** is the implicit route with a tight solve, checked once against a long
unroll. Exercise 3.4's table is that check.

### 5.3 Kinks in practice

Every ReLU network is differentiated at kinks constantly, and it works. Here is why, and what could
go wrong.

**Why it works.** The set where a ReLU's argument is exactly zero is a measure-zero subset of the
input space, and a randomly initialized network evaluated on real data lands there with probability
zero in exact arithmetic. In floating point the set is not measure zero, it is the finitely many
representable values that hit zero exactly, but it is still extraordinarily rare.

More importantly, the derivative returned off the kink is **correct**, and the function is
differentiable almost everywhere. So the gradient used by training is the true gradient at almost
every point, and stochastic gradient descent averages over points anyway.

**What could go wrong, three ways.**

*A structurally exact zero.* Padding a convolution with zeros, masking a sequence, or a
zero-initialized bias makes many pre-activations exactly $0$ **by construction**, not by accident. The
measure-zero argument does not apply, because the value is not random. Whether that matters depends
on which subgradient the framework returns, and frameworks disagree: some return $0$ for
`relu'(0)` and some return $1$.

*A max over a tie.* `max_pool` and `argmax` break ties by index, so a tie sends the whole gradient to
one input and none to the other. With quantized or low-precision activations ties are common, so the
gradient becomes an artifact of the tie-breaking rule.

*Convergence to a kink.* An optimizer can drive a parameter towards a point where a kink is active,
because the kink is often where the loss is minimized. Then the iterates approach the bad set rather
than avoiding it, and the measure-zero argument is about a fixed point, not about a limit.

**How to detect it.** Count the exact zeros. Instrument the forward pass to record the fraction of
pre-activations that are exactly $0.0$, and watch it over training. On a healthy run it is a
handful; a large or growing fraction means the argument above has stopped applying and the gradients
are subgradients chosen by the source code.

The cheap second check is lesson 97's own experiment: compute the same gradient with `relu` written
two different ways and compare. If they differ, the difference is exactly the set of active kinks.

---

## Lesson 98, Scientific Machine Learning and Inverse Problems

### 1.1 What makes a problem ill-posed

Hadamard's three conditions for a well-posed problem are that a solution **exists**, that it is
**unique**, and that it **depends continuously on the data**. A problem failing any of them is
ill-posed, and for the inverse problems in this lesson it is the third that fails.

**The property of the operator that causes it: the forward map smooths.** Integrating against a
kernel $k$,

$$
(Ax)(s) = \int k(s - t)\,x(t)\,dt ,
$$

damps high frequencies, and damps them more the higher they are. In singular value terms the
singular values $\sigma_i$ decay to zero, geometrically for an analytic kernel. The inverse
multiplies by $1/\sigma_i$, which is unbounded, so an arbitrarily small change in $b$ can produce an
arbitrarily large change in $x$.

**Two things this is not.** It is not a property of the discretization: every discretization inherits
it, and lesson 98's section 3 measured the condition number passing $10^{17}$ at $64$ grid points and
growing. And it is not a bug: the continuous operator genuinely has no bounded inverse, so no
algorithm can compute one.

The operator being compact is the general statement. A compact operator on an infinite dimensional
space has singular values tending to zero, so its inverse is unbounded, and every Fredholm equation
of the first kind with a continuous kernel has a compact operator.

### 1.2 Why refining the grid does not help

Take the singular value decomposition of the discretized operator at grid size $n$. Refining to $2n$
adds $n$ new singular values, and they are added **at the bottom**: the spectrum decays like
$c\,r^i$, so the new ones are smaller than every existing one.

Three consequences, and the third is the one that matters.

**Every new direction is below the noise.** A direction is usable only while $\sigma_i$ exceeds the
noise level divided by the size of the answer, which is exercise 1.3's Picard condition. The new
directions have the smallest $\sigma_i$, so they are the first to fail that test.

**The condition number grows without bound.** $\kappa = \sigma_1/\sigma_n$ and $\sigma_n$ falls
geometrically, so $\kappa$ grows geometrically in $n$. Lesson 98's section 3 measured it passing
$1/\varepsilon = 4.5\times10^{15}$ at $64$ points.

**So the extra unknowns are unconstrained.** Lesson 98 measured this directly: going from $128$ to
$256$ grid points adds $128$ unknowns and $3$ usable singular values. The other $125$ directions are
determined by nothing at all, and a solver that tries to determine them is fitting rounding.

**The contrast that makes the point.** The second difference operator on the same grid has
$\kappa \sim n^2$, fitted at $n^{1.975}$, and reaches only $26768$ at $256$ points. A forward problem
gets harder polynomially, which is manageable; an inverse problem gets harder geometrically, which is
not.

### 1.3 The discrete Picard condition

Write the data in the left singular basis. The least squares solution is

$$
x = \sum_i \frac{u_i^{\mathsf T}b}{\sigma_i}\,v_i .
$$

The **discrete Picard condition** is that the coefficients $|u_i^{\mathsf T}b|$ decay faster than the
singular values $\sigma_i$, so the ratios $|u_i^{\mathsf T}b|/\sigma_i$ stay bounded and the sum
converges to something sensible.

**What it is used for: counting the information.** Noise is white, so its coefficients
$|u_i^{\mathsf T}\eta|$ sit at a roughly constant level $\tau$, independent of $i$. The signal's
coefficients decay. So the measured $|u_i^{\mathsf T}b|$ decays until it reaches $\tau$ and then
flattens. **The index where it flattens is the number of components the data contains**, and every
component past it contributes $\tau/\sigma_i$, a constant divided by a tiny number.

Lesson 98's section 4 measured it: $31$ of $128$ components usable at a noise level of $10^{-4}$,
$22$ at $10^{-2}$, and the plain pseudoinverse wrong by $5.8\times10^{7}$.

**The practical use is a plot.** Put $\sigma_i$, $|u_i^{\mathsf T}b|$ and their ratio on one log
axis. If the coefficients never fall below the singular values, the problem is not solvable at that
noise level and no regularization will rescue it. If they do, the crossing gives a starting
regularization parameter: $\lambda \approx \sigma_k^2$ at the crossing index $k$.

**And the top row of lesson 98's table.** With no added noise at all the count is $61$ of $128$, not
$128$, because the operator is stored in floating point and $\varepsilon$ is a noise level. **There
is no such thing as exact data.**

### 1.4 What early stopping regularizes, and its parameter

Conjugate gradient on $A^{\mathsf T}Ax = A^{\mathsf T}b$ builds its iterate in the Krylov space
$\text{span}\{A^{\mathsf T}b, (A^{\mathsf T}A)A^{\mathsf T}b, \dots\}$. After $k$ steps the iterate
is $x_k = p_k(A^{\mathsf T}A)A^{\mathsf T}b$ for some polynomial $p_k$ of degree $k-1$, which in the
singular basis is a **filter**:

$$
x_k = \sum_i \phi_k(\sigma_i^2)\,\frac{u_i^{\mathsf T}b}{\sigma_i}\,v_i ,
\qquad \phi_k(\mu) = \mu\,p_k(\mu) .
$$

Conjugate gradient minimizes the residual over that space, so it fits the largest $\sigma_i$ first,
and $\phi_k$ is near $1$ for large $\sigma$ and near $0$ for small $\sigma$ with the transition
moving down as $k$ grows.

**So the iteration count is the regularization parameter**, playing exactly the role $\lambda$ plays
in Tikhonov. Stopping early keeps a filter that has not yet reached the noisy directions.

Lesson 98's section 7 measured all of it. The error falls to $0.1239$ at iteration $60$ and rises to
$14.6$ by $400$, a factor of $118$. The best iteration moves from $98$ to $7$ as the noise goes from
$10^{-5}$ to $10^{-2}$, exactly as the best $\lambda$ moves up. And at its best, conjugate gradient
beats the best Tikhonov solution by $0.9$ per cent, so it is not an approximation to regularization,
it is regularization.

**The name for the shape is semiconvergence**, and the practical rule it implies is the opposite of
every other solver in this course: **stop before it converges.**

### 1.5 The two routes to a neural ODE gradient

**Discretize then optimize.** Run the solver, record every state and every arithmetic operation, and
differentiate the resulting program with reverse mode. The answer is the exact gradient of the
**discrete** function that was computed. Memory is the tape, proportional to the number of steps.

**Optimize then discretize.** Write down the adjoint equation for the continuous problem,

$$
\frac{da}{dt} = -a^{\mathsf T}\frac{\partial f}{\partial z} ,
\qquad
\frac{\partial L}{\partial\theta} = -\int_0^1 a^{\mathsf T}\frac{\partial f}{\partial\theta}\,dt ,
$$

and solve it numerically backwards in time. The answer is a **discretization of the exact gradient**
of the continuous problem. Memory is one state and one adjoint, independent of the step count.

**The property that distinguishes them: they are gradients of different functions**, and they agree
only in the limit $h \to 0$. Lesson 98's section 9 measured the gap at $2.2\times10^{-2}$ at $16$
steps and $6.8\times10^{-4}$ at $512$, falling like $h^{0.998}$, with a memory ratio reaching $342$.

**Which to want.** Usually the discrete one, because the discrete function is what the training loop
evaluates and reports, so its gradient is the one that makes the reported loss fall monotonically.
The adjoint is wanted when the memory makes the alternative impossible, which is the situation neural
ODEs were introduced to address.

### 2.1 Why an analytic kernel gives a geometric spectrum

Let $k$ be analytic, so it extends holomorphically to a strip of width $2a$ around the real axis.
Then its Fourier transform decays exponentially: $|\hat k(\xi)| \le Ce^{-a|\xi|}$. That is a standard
Paley-Wiener statement and it is the whole content of the result.

For a **convolution** operator $Ax = k * x$ on a periodic domain, the eigenfunctions are the Fourier
modes and the eigenvalues are $\hat k(\xi_i)$ at the discrete frequencies $\xi_i \sim i$. So

$$
\sigma_i = |\hat k(\xi_i)| \le C e^{-a i} = C r^{i} , \qquad r = e^{-a} < 1 ,
$$

which is geometric decay with a rate set by the width of the strip of analyticity.

For a Gaussian kernel of width $w$, $\hat k(\xi) = e^{-w^2\xi^2/2}$, so the decay is even faster than
geometric: $\sigma_i \sim e^{-w^2 i^2/2}$. Over the first few dozen indices that is indistinguishable
from geometric with a rate that slowly steepens, which is what lesson 98's section 3 measured at
$0.657$ per index and nearly constant across grid sizes.

**The general statement, for a non-convolution kernel**, is that the singular values of an integral
operator with a $C^p$ kernel decay at least like $i^{-p}$, and faster than any power for an analytic
one. So the smoothness of the kernel is the ill-posedness: **the better behaved the forward problem,
the worse the inverse one.**

### 2.2 Tikhonov filters and truncation as their limit

From lesson 94's exercise 2.1, the Tikhonov solution with $L = I$ is

$$
x_\lambda = \sum_i \frac{\sigma_i^2}{\sigma_i^2+\lambda}\,\frac{u_i^{\mathsf T}b}{\sigma_i}\,v_i
= \sum_i f_i(\lambda)\,\frac{u_i^{\mathsf T}b}{\sigma_i}\,v_i .
$$

The truncated SVD solution keeping $k$ components is

$$
x_k = \sum_{i \le k} \frac{u_i^{\mathsf T}b}{\sigma_i}\,v_i ,
\qquad
f_i^{\text{trunc}} = \mathbb{1}[i \le k] .
$$

**Truncation as a limit of a sharpening family.** Generalize the filter to

$$
f_i^{(p)}(\lambda) = \frac{\sigma_i^{2p}}{\sigma_i^{2p} + \lambda^{p}} ,
$$

which is Tikhonov at $p = 1$. Write $\sigma_i^2 = \lambda\,e^{s}$; then

$$
f_i^{(p)} = \frac{e^{ps}}{e^{ps}+1} = \frac{1}{1 + e^{-ps}} ,
$$

a logistic function of $s = \log(\sigma_i^2/\lambda)$ with steepness $p$. As $p \to \infty$ this
tends to $\mathbb{1}[s > 0] = \mathbb{1}[\sigma_i^2 > \lambda]$, which is the truncation filter with
the cut at $\sigma_k^2 = \lambda$.

**So the two are the same one-parameter family at two steepnesses**, $p = 1$ and $p = \infty$, and
choosing between them is choosing how sharply to treat the directions near the cut. Lesson 98's
section 6 measured them within a factor of $1.05$ of each other on the same data, which is what a
family with the same cut and a different edge should give.

### 2.3 The adjoint equation from a Lagrangian

Minimize $L(z(T))$ subject to $\dot z = f(z,\theta)$, $z(0) = z_0$. Introduce a multiplier function
$a(t)$ and form

$$
\mathcal{L} = L(z(T)) - \int_0^T a(t)^{\mathsf T}\left(\dot z - f(z,\theta)\right)dt .
$$

The constraint holds, so $\mathcal{L} = L$ for any $a$. Perturb $\theta$ by $\delta\theta$, which
perturbs $z$ by $\delta z$ with $\delta z(0) = 0$:

$$
\delta\mathcal{L} = \nabla L(z(T))^{\mathsf T}\delta z(T)
- \int_0^T a^{\mathsf T}\left(\delta\dot z - \frac{\partial f}{\partial z}\delta z
- \frac{\partial f}{\partial\theta}\delta\theta\right)dt .
$$

Integrate the $a^{\mathsf T}\delta\dot z$ term by parts:

$$
\int_0^T a^{\mathsf T}\delta\dot z\,dt
= \left[a^{\mathsf T}\delta z\right]_0^T - \int_0^T \dot a^{\mathsf T}\delta z\,dt
= a(T)^{\mathsf T}\delta z(T) - \int_0^T \dot a^{\mathsf T}\delta z\,dt ,
$$

using $\delta z(0) = 0$. Substituting and collecting the $\delta z$ terms:

$$
\delta\mathcal{L}
= \left(\nabla L(z(T)) - a(T)\right)^{\mathsf T}\delta z(T)
+ \int_0^T\left(\dot a + \frac{\partial f}{\partial z}^{\mathsf T}a\right)^{\mathsf T}\delta z\,dt
+ \int_0^T a^{\mathsf T}\frac{\partial f}{\partial\theta}\,\delta\theta\,dt .
$$

**Now choose $a$ to kill the terms involving $\delta z$**, which is the whole trick, because
$\delta z$ is expensive to compute and $a$ is not:

$$
\boxed{\ a(T) = \nabla L(z(T)) , \qquad \dot a = -\frac{\partial f}{\partial z}^{\mathsf T}a . \ }
$$

That is a terminal value problem, integrated backwards. What remains is

$$
\frac{\partial L}{\partial\theta} = \int_0^T a(t)^{\mathsf T}\frac{\partial f}{\partial\theta}\,dt .
$$

**Memory is one state and one adjoint**, because nothing in either equation refers to more than the
current time. That is the property the method exists for, and lesson 98's section 9 measured it at
$18$ numbers against unrolling's $6156$ at $512$ steps.

### 2.4 Conjugate gradient as a polynomial filter

After $k$ steps starting from $x_0 = 0$, conjugate gradient's iterate lies in the Krylov space

$$
\mathcal{K}_k = \text{span}\{r_0,\ Mr_0,\ \dots,\ M^{k-1}r_0\} ,
\qquad M = A^{\mathsf T}A , \quad r_0 = A^{\mathsf T}b ,
$$

so $x_k = q_{k-1}(M)\,A^{\mathsf T}b$ for some polynomial $q_{k-1}$ of degree $k-1$. In the singular
basis, $M v_i = \sigma_i^2 v_i$ and $A^{\mathsf T}b = \sum_i \sigma_i(u_i^{\mathsf T}b)v_i$, so

$$
x_k = \sum_i q_{k-1}(\sigma_i^2)\,\sigma_i\,(u_i^{\mathsf T}b)\,v_i
= \sum_i \underbrace{\sigma_i^2 q_{k-1}(\sigma_i^2)}_{\phi_k(\sigma_i^2)}\,
\frac{u_i^{\mathsf T}b}{\sigma_i}\,v_i .
$$

**That is a filter, with $\phi_k(0) = 0$ for every $k$**, since $\phi_k(\mu) = \mu q_{k-1}(\mu)$.

**Why the early iterates are smooth.** Conjugate gradient chooses $q$ to minimize the $M$-norm of the
error, which is $\sum_i (1 - \phi_k(\sigma_i^2))^2\,\sigma_i^2\,\alpha_i^2$ with $\alpha_i$ the true
coefficients. Each term is weighted by $\sigma_i^2$, so the minimization cares most about the
directions with the largest $\sigma_i$, and a degree $k-1$ polynomial has only $k$ roots to spend.
It spends them where the weight is: the large singular values. So $\phi_k \approx 1$ there and
$\phi_k \approx 0$ elsewhere, which is a low-pass filter and hence a smooth iterate.

**And why the later ones are not.** As $k$ grows, the polynomial has enough degrees of freedom to
reach the small $\sigma_i$ too, and there $u_i^{\mathsf T}b$ is noise. That is the rise in lesson
98's section 7, and the reason the best $k$ falls as the noise grows: more noise means the crossover
index is smaller, so fewer roots can be spent before the polynomial reaches it.

### 2.5 The discrepancy principle

Morozov's discrepancy principle chooses the **largest** $\lambda$ whose residual is still consistent
with the noise:

$$
\lambda^\star = \max\left\{\lambda : \lVert Ax_\lambda - b\rVert \le \tau\,\delta\right\} ,
$$

where $\delta$ is the noise level, $\lVert\eta\rVert \approx \delta$, and $\tau \gtrsim 1$ is a
safety factor.

**The reasoning.** The data is $b = Ax^\star + \eta$. Any candidate $x$ with
$\lVert Ax - b\rVert < \delta$ is fitting the noise, because the true solution itself has residual
$\delta$. Any candidate with $\lVert Ax - b\rVert \gg \delta$ is not fitting the data. So the right
answer sits at $\lVert Ax - b\rVert \approx \delta$, and among all such candidates the most
regularized one is the safest.

The residual $\lVert Ax_\lambda - b\rVert$ is continuous and increasing in $\lambda$, from
$\lVert (I - AA^{+})b\rVert$ at $\lambda = 0$ to $\lVert b\rVert$ as $\lambda \to \infty$, so the
equation $\lVert Ax_\lambda - b\rVert = \tau\delta$ has a unique solution whenever $\tau\delta$ is in
that range, and a bisection finds it.

**What it needs that the L-curve does not: the noise level $\delta$.** That is the entire difference.
When $\delta$ is known, from the instrument or from repeated measurements, the discrepancy principle
is the sharpest of the three selectors and has convergence guarantees the others lack. When it is
not, the principle cannot be applied at all, and the L-curve and generalized cross-validation exist
precisely for that case.

**And $\tau$ matters.** With $\tau = 1$ the method is known to under-regularize; the standard advice
is $\tau$ between $1.1$ and $1.5$, and exercise 3.1 measures the sensitivity.

### 3.1 Generalized cross-validation against the other two selectors

```python
import numpy as np
from nalib import sciml


def gcv_penalty(problem, data, penalties):
    """Minimize n ||A x_lam - b||^2 / (n - sum f_i)^2, which needs neither noise nor answer."""
    left, s, _ = np.linalg.svd(problem["matrix"], full_matrices=False)
    coefficients = left.T @ np.asarray(data, dtype=float)
    rows = problem["matrix"].shape[0]
    scores = []
    for lam in penalties:
        filters = s ** 2 / (s ** 2 + lam)
        residual = float(np.sum(((1.0 - filters) * coefficients) ** 2))
        scores.append(rows * residual / (rows - float(np.sum(filters))) ** 2)
    return float(penalties[int(np.argmin(scores))]), np.array(scores)


def discrepancy_penalty(problem, data, penalties, level, safety=1.2):
    """The largest penalty whose residual still matches the noise, which needs the noise."""
    wanted = safety * level * np.sqrt(np.asarray(data).size)
    chosen = float(penalties[0])
    for lam in sorted(penalties, reverse=True):
        if sciml.tikhonov(problem, data, float(lam))["residual"] <= wanted:
            chosen = float(lam)
            break
    return chosen


size = 96
problem = sciml.blur(size)
truth = sciml.test_signal(problem["grid"])
grid = np.geomspace(1e-12, 1e2, 61)
scale = float(np.linalg.norm(truth))

print(f"{'noise':>9}{'oracle':>12}{'l-curve':>12}{'gcv':>12}{'discrepancy':>14}"
      f"{'  |  errors, relative to the oracle':>36}")
for level in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
    made = sciml.noisy_data(problem, truth, level=level, seed=42)
    curve = sciml.l_curve(problem, made["data"], grid, truth=truth)
    gcv, _ = gcv_penalty(problem, made["data"], grid)
    morozov = discrepancy_penalty(problem, made["data"], grid, level)
    errors = {}
    for name, lam in (("oracle", curve["oracle_penalty"]), ("l-curve", curve["corner_penalty"]),
                      ("gcv", gcv), ("discrepancy", morozov)):
        solution = sciml.tikhonov(problem, made["data"], lam)["solution"]
        errors[name] = float(np.linalg.norm(solution - truth) / scale)
    print(f"{level:>9.0e}{curve['oracle_penalty']:>12.2e}"
          f"{curve['corner_penalty']:>12.2e}{gcv:>12.2e}{morozov:>14.2e}"
          f"  |  " + "  ".join(f"{name} {errors[name] / errors['oracle']:.3f}"
                               for name in ("l-curve", "gcv", "discrepancy")))
```

**The penalties disagree by orders of magnitude and the errors do not.** That is the practical
finding and it repeats lesson 98's section 5: the error curve is flat near its minimum, so a selector
that lands within an order of magnitude of the oracle penalty lands within a few per cent of the
oracle error.

**Their failure modes differ, and that is how to choose.** The discrepancy principle needs $\delta$
and is the sharpest when it has it. GCV needs nothing and degrades when the score becomes flat, which
happens at very low noise. The L-curve needs nothing and degrades when the curve has no clear corner,
which happens when the Picard condition is badly violated.

### 3.2 Total variation regularization

Tikhonov with $L = D_2$ penalizes $\lVert D_2 x\rVert_2^2$, which is smooth and therefore smooths.
Total variation penalizes $\lVert D_1 x\rVert_1$, which is not smooth, and preserves jumps. Lesson
90's proximal gradient method solves it.

```python
import numpy as np
from nalib import sciml


def first_difference(size):
    matrix = np.zeros((size - 1, size))
    index = np.arange(size - 1)
    matrix[index, index] = -1.0
    matrix[index, index + 1] = 1.0
    return matrix


def total_variation(problem, data, weight, steps=4000, seed=0):
    """Minimize ||Ax - b||^2/2 + weight * ||D1 x||_1 by proximal gradient on the dual.

    The dual of the total variation term is a projection onto a box, which is what makes this
    tractable: the nonsmooth penalty becomes a constraint that has a closed form projection.
    """
    a = problem["matrix"]
    b = np.asarray(data, dtype=float)
    d = first_difference(a.shape[1])
    gram = a.T @ a
    rhs = a.T @ b
    step = 1.0 / (float(np.linalg.eigvalsh(gram)[-1]) + 8.0 * weight)
    x = np.linalg.lstsq(a, b, rcond=None)[0] * 0.0
    dual = np.zeros(d.shape[0])
    for _ in range(steps):
        x = x - step * (gram @ x - rhs + weight * (d.T @ dual))
        dual = np.clip(dual + step * 8.0 * (d @ x), -1.0, 1.0)
    return x


size = 96
problem = sciml.blur(size)
grid = np.geomspace(1e-12, 1e2, 41)
print(f"{'signal':>9}{'best Tikhonov L=I':>21}{'best Tikhonov L=D2':>22}"
      f"{'best total variation':>23}")
for kind in ("smooth", "box", "mixed"):
    truth = sciml.test_signal(problem["grid"], kind)
    made = sciml.noisy_data(problem, truth, level=1e-4, seed=42)
    scale = float(np.linalg.norm(truth))
    identity = min(float(np.linalg.norm(sciml.tikhonov(problem, made["data"], lam,
                                                       order=0)["solution"] - truth) / scale)
                   for lam in grid)
    smoothness = min(float(np.linalg.norm(sciml.tikhonov(problem, made["data"], lam,
                                                         order=2)["solution"] - truth) / scale)
                     for lam in grid)
    variation = min(float(np.linalg.norm(total_variation(problem, made["data"], w) - truth)
                          / scale) for w in (1e-4, 1e-3, 1e-2, 1e-1))
    print(f"{kind:>9}{identity:>21.6f}{smoothness:>22.6f}{variation:>23.6f}")
```

**The comparison is a comparison of priors, not of algorithms.** $L = I$ says the answer is small,
$L = D_2$ says it is smooth, and total variation says it is **piecewise** smooth, with a few jumps
allowed. The box signal has one jump and nothing else, so it is the case where the third prior is
right and the first two are wrong.

**And the reason total variation is harder to use** is visible in the code: the penalty is not
differentiable, so there is no linear solve. Lesson 90's proximal machinery is needed, the answer
depends on the number of iterations, and there is no analogue of the L-curve because there is no
filter factor formula. That is the price of a prior that fits.

### 3.3 Two-dimensional deblurring, matrix free

A separable blur in two dimensions is a Kronecker product, $A = A_y \otimes A_x$, so a matrix-vector
product is two small matrix multiplies and the big matrix is never formed.

```python
import numpy as np
from nalib import sciml

side = 64
one_dimensional = sciml.blur(side, width=0.06)["matrix"]
rng = np.random.default_rng(3)
image = np.zeros((side, side))
image[side // 4:side // 2, side // 4:side // 2] = 1.0
row, column = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
image += 0.7 * np.exp(-((row - 3 * side // 4) ** 2 + (column - 3 * side // 4) ** 2) / 40.0)


def forward(x):
    """(Ay kron Ax) applied to a vectorized image, as two small matrix products."""
    square = x.reshape(side, side)
    return (one_dimensional @ square @ one_dimensional.T).ravel()


def normal(x, penalty):
    return forward(forward(x)) + penalty * x


blurred = forward(image.ravel())
noisy = blurred + 1e-4 * rng.standard_normal(blurred.size)
scale = float(np.linalg.norm(image))

print(f"the operator would be {side * side} by {side * side} = "
      f"{(side * side) ** 2} entries if it were formed")
print(f"it is stored as one {side} by {side} matrix: {side * side} entries, "
      f"a factor of {(side * side) ** 2 / (side * side):.0f}")
print(f"\n{'penalty':>10}{'cg iterations':>16}{'relative error':>17}")
for penalty in (1e-6, 1e-4, 1e-2, 1e-1):
    run = sciml.conjugate_gradient(lambda v, p=penalty: normal(v, p),
                                   forward(noisy), 200, truth=image.ravel())
    errors = [entry["error"] for entry in run["history"]]
    best = int(np.argmin(errors))
    print(f"{penalty:>10.0e}{best + 1:>16}{errors[best]:>17.6f}")

for cap in (300, 1500, 4000):
    run = sciml.conjugate_gradient(lambda v: normal(v, 0.0), forward(noisy), cap,
                                   truth=image.ravel())
    errors = [entry["error"] for entry in run["history"]]
    best = int(np.argmin(errors))
    print(f"\nno penalty, {len(errors)} iterations: best is number {best + 1} at "
          f"{errors[best]:.6f}, the last is {errors[-1]:.6f}, "
          f"a rise of {errors[-1] / errors[best]:.3f}")
```

**The memory saving is the square of the side length**, which is the whole reason Kronecker structure
is exploited. At $64 \times 64$ the assembled operator would have $1.7\times10^{7}$ entries and the
factored one has $4096$.

**And the last block is section 7 in two dimensions, with a caveat the one-dimensional version did
not need.** At $300$ iterations the error is still falling, so the run has not turned yet and a
report stopping there would have concluded, wrongly, that semiconvergence does not happen here. It
does: the best iterate is number $637$, the error has risen by $1.215$ by iteration $1500$ and by
$3.009$ by $4000$. It simply takes longer, because the two-dimensional operator is a Kronecker
product whose spectrum falls more slowly per index than its one-dimensional factor's.

That is worth naming as a trap. **Semiconvergence is not visible until the iteration reaches the
noisy directions**, and how many iterations that takes is a property of the spectrum, not a universal
number. A run cut off before the turn looks like a converging solver.

The penalty column shows the other half: adding a penalty reduces the iterations needed and raises
the error floor, which are the two sides of the same filter. The usual practice is a small penalty
for conditioning and the iteration count for regularization, and this table is why: at
$\lambda = 10^{-1}$ the run stops in $41$ iterations and is $1.5$ times worse than at $10^{-6}$.

### 3.4 A nonlinear boundary value problem

Redo lesson 98's section 8 comparison where the linear basis also needs Newton's method, so the
"one linear solve" advantage is gone.

```python
import numpy as np
from nalib import sciml

problem = sciml.boundary_value_problem()
points = 40
inside = np.linspace(0.0, 1.0, points + 2)[1:-1]
fine = np.linspace(0.0, 1.0, 401)
ends = np.array([0.0, 1.0])
strength = 4.0


def exact(x):
    return problem["exact"](x)


def source(x):
    """The forcing that makes u'' + strength * u^3 = f have the same exact solution."""
    return problem["source"](x) + strength * exact(x) ** 3


def residual(coefficients, basis, edge):
    u = basis["values"] @ coefficients
    return np.concatenate([basis["second"] @ coefficients + strength * u ** 3 - source(inside),
                           10.0 * (edge @ coefficients
                                   - np.array([problem["left"], problem["right"]]))])


def jacobian(coefficients, basis, edge):
    u = basis["values"] @ coefficients
    top = basis["second"] + 3.0 * strength * (u ** 2)[:, None] * basis["values"]
    return np.vstack([top, 10.0 * edge])


print(f"u'' + {strength:g} u^3 = f, solved by Newton on the collocation system")
print(f"{'terms':>7}{'newton steps':>15}{'final residual':>18}{'error':>14}")
for terms in (8, 12, 16, 20, 24):
    basis = sciml.chebyshev_basis(inside, terms)
    edge = sciml.chebyshev_basis(ends, terms)["values"]
    coefficients = np.zeros(terms)
    previous = np.inf
    settled = 0
    for steps in range(1, 31):
        norm = float(np.linalg.norm(residual(coefficients, basis, edge)))
        if norm >= 0.5 * previous:
            settled = steps
            break
        previous = norm
        coefficients = coefficients - np.linalg.lstsq(jacobian(coefficients, basis, edge),
                                                      residual(coefficients, basis, edge),
                                                      rcond=None)[0]
        settled = steps
    approximation = sciml.chebyshev_basis(fine, terms)["values"] @ coefficients
    error = float(np.linalg.norm(approximation - exact(fine)) / np.linalg.norm(exact(fine)))
    print(f"{terms:>7}{settled:>15}{previous:>18.3e}{error:>14.4e}")

print(f"\nlesson 98's linear problem needed 1 solve; this one needs a handful,")
print(f"and the trained network of section 8 needed 111000 function evaluations")
```

**The nonlinearity costs three to five linear solves and does not change the conclusion.** Newton's
residual falls to $7.6\times10^{-11}$ in five steps at $24$ terms, and the spectral convergence in
the term count survives: $9.9\times10^{-1}$, $1.5\times10^{-2}$, $2.4\times10^{-6}$,
$1.3\times10^{-8}$, $1.7\times10^{-11}$. So the linear basis still reaches accuracies the trained
fit of section 8 does not approach, now at five solves instead of one against $111000$ function
evaluations.

**What would change the conclusion** is a problem where the Newton iteration itself fails: a strong
nonlinearity with multiple solutions, or a solution with a moving front that a global polynomial
basis cannot represent. Those are real cases, and they are the ones where a flexible basis has
something to offer. This problem is not one of them, and neither was section 8's.

### 3.5 The adjoint with a checkpointed forward pass

Three points on one curve: store nothing and recompute, store everything, or store some.

```python
import math
import numpy as np
from nalib import sciml

problem = sciml.flow_problem(3, seed=42)
theta = problem["parameters"]
dimension = problem["dimension"]

print(f"{'steps':>8}{'adjoint memory':>17}{'checkpointed':>15}{'unrolled memory':>18}"
      f"{'gap to unrolled':>18}")
for steps in (64, 256, 1024):
    adjoint = sciml.adjoint_gradient(problem, theta, steps)
    unrolled = sciml.unrolled_gradient(problem, theta, steps)
    gap = float(np.linalg.norm(adjoint["gradient"] - unrolled["gradient"])
                / np.linalg.norm(unrolled["gradient"]))
    gap_size = max(int(round(math.sqrt(steps))), 1)
    checkpointed = dimension * (steps // gap_size + gap_size) + theta.size
    print(f"{steps:>8}{adjoint['memory']:>17}{checkpointed:>15}"
          f"{unrolled['memory']:>18}{gap:>18.3e}")

print(f"\nthe three schemes hold O(1), O(sqrt(n)) and O(n) states")
print(f"and the adjoint's answer is a different gradient, not a cheaper one")
```

**The three schemes are the ends and the middle of lesson 97's exercise 2.4.** The adjoint holds
$O(1)$ by recomputing or re-integrating the forward trajectory; unrolling holds $O(n)$; checkpointing
holds $O(\sqrt n)$ and is what a practical neural ODE implementation actually does.

**And the last line is the caveat that matters.** The adjoint is not a memory-efficient way to
compute the unrolled gradient. It computes a **different** gradient, and the gap between them falls
only as the step size does. Checkpointed **unrolling**, by contrast, gives the discrete gradient
exactly at $O(\sqrt n)$ memory, which is often the better trade and is what lesson 97's section 7
measured.

### 4.1 How the best penalty scales with the noise

Lesson 94's exercise 2.4 derived $\lambda^\star \sim \tau^2$ for a fixed spectrum. Here it is over
six decades on the blur.

```python
import numpy as np
from nalib import sciml

problem = sciml.blur(96)
truth = sciml.test_signal(problem["grid"])
grid = np.geomspace(1e-16, 1e2, 73)
scale = float(np.linalg.norm(truth))
repeats = 5

print(f"{'noise':>10}{'best lambda':>15}{'best error':>14}{'lambda / noise^2':>19}"
      f"{'lambda / noise':>17}")
rows = []
for level in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
    chosen, best = [], []
    for trial in range(repeats):
        made = sciml.noisy_data(problem, truth, level=level, seed=100 + trial)
        errors = [float(np.linalg.norm(sciml.tikhonov(problem, made["data"],
                                                      lam)["solution"] - truth) / scale)
                  for lam in grid]
        index = int(np.argmin(errors))
        chosen.append(grid[index])
        best.append(errors[index])
    lam = float(np.median(chosen))
    rows.append((level, lam))
    print(f"{level:>10.0e}{lam:>15.3e}{np.median(best):>14.6f}"
          f"{lam / level ** 2:>19.3e}{lam / level:>17.3e}")

slope = np.polyfit(np.log([r[0] for r in rows]), np.log([r[1] for r in rows]), 1)[0]
print(f"\nfitted: lambda ~ noise to the {slope:.4f}")
print(f"lesson 94's exercise 2.4 predicts 2 for a fixed spectrum")
```

**The fit lands at $1.979$ against a predicted $2$**, over six decades of noise, which confirms
exercise 2.4 of lesson 94: the optimal penalty per component is $\tau^2/\alpha_i^2$ and the
$\alpha_i$ here are of order one relative to each other.

**Two things the table shows that the exponent hides.** The `lambda / noise^2` column is not
constant, it wanders between $31$ and $562$, which is the sampling noise of taking a median over five
seeds on a grid of penalties spaced by $\sqrt{10}$. And the **error** barely moves: from $0.106$ at a
noise level of $10^{-7}$ to $0.169$ at $10^{-2}$, a factor of $1.6$ over five decades of noise. The
reconstruction is limited by the operator, not by the data.

**When the exponent would not be $2$.** If the signal's coefficients decayed geometrically like the
singular values, the per-component optima would be spread over many decades and the sum's minimizer
would be pulled towards the components dominating the error, giving an exponent between $1$ and $2$.
The classical result covers that case: $\lambda \sim \tau^{4/(2\mu+1)}$ for a source condition of
order $\mu$, which is $2$ at $\mu = 1/2$ and less below it.

### 4.2 Where a trained basis actually starts to win

Section 8 was one dimensional, where a tensor product basis is trivial. The claim usually made for
physics-informed networks is about high dimension, so this is the measurement that could reverse it.

First the counting argument, which is the part everybody makes.

```python
import numpy as np
from math import comb
from nalib import sciml

print("how many coefficients each basis needs on the unit cube")
print(f"{'d':>4}{'tensor at m=20':>18}{'tensor at m=12':>18}{'sparse at level 6':>21}"
      f"{'network, 64 units':>21}{'tensor memory (GB)':>21}")
for dimension in (1, 2, 3, 4, 6, 8, 12):
    grid = 20 ** dimension
    smaller = 12 ** dimension
    sparse = comb(6 + dimension, dimension)
    network = 64 * (dimension + 2)
    print(f"{dimension:>4}{grid:>18.3e}{smaller:>18.3e}{sparse:>21}"
          f"{network:>21}{grid * 8 / 1e9:>21.3e}")

print(f"\na machine with 64 GB holds a tensor grid of "
      f"{int((64e9 / 8) ** (1 / 4))} per side in 4 dimensions "
      f"and {int((64e9 / 8) ** (1 / 8))} per side in 8")
print("a sparse grid at level 6 stays under a thousand coefficients through 8 dimensions")
```

**The counting says the tensor product dies and says nothing about the winner**, because the
classical competitor is not a tensor product. A sparse grid needs $C(\text{level}+d, d)$
coefficients, which is polynomial in $d$, and it stays under a thousand through eight dimensions. So
the memory argument rules out one classical method and leaves another standing.

Now the measurement, at a matched parameter budget.

```python
from nalib import sciml

out = sciml.where_a_trained_basis_starts_to_win()
for budget in sorted(out["crossovers"]):
    print(f"\nbudget {budget} parameters, {out['steps']} training steps, "
          f"per axis degree capped at {out['max_degree']}")
    print(f"{'d':>4}{'level':>7}{'basis':>8}{'sparse grid':>14}{'units':>7}"
          f"{'trained net':>14}{'frozen net':>13}{'grid / net':>13}{'train gain':>13}")
    for row in out["rows"]:
        if row["budget"] != budget:
            continue
        print(f"{row['dimension']:>4}{row['level']:>7}{row['grid_size']:>8}"
              f"{row['grid_error']:>14.3e}{row['units']:>7}{row['trained_error']:>14.3e}"
              f"{row['frozen_error']:>13.3e}{row['ratio']:>13.3g}"
              f"{row['training_gain']:>13.3g}")
    print(f"  crossover at d = {out['crossovers'][budget]}, usable to d = "
          f"{out['usable_to'][budget]['grid']} for the grid and "
          f"{out['usable_to'][budget]['trained']} for the network")

print(f"\nlargest classical win {out['largest_classical_win']:.3g}")
print(f"largest trained win   {out['largest_trained_win']:.3g}")
print(f"worst error past the crossover {out['worst_error_past_the_crossover']:.3g}, "
      f"against a usable threshold of {out['usable_threshold']}")

assert out["the_crossover_agrees_across_budgets"]
assert out["neither_is_usable_past_the_crossover"]
```

**The crossover is at $d = 4$, and the counting argument put it past $d = 6$.** That is a
correction to the obvious reasoning and it is worth being explicit about why the reasoning failed.
The memory table compares a network against a **tensor product**, which is the worst classical
method, and concludes that the network survives where it does not. Against a sparse grid, which is
what a numerical analyst would actually use, the classical method survives much further than the
tensor count suggests and still loses earlier than the naive comparison predicted. Two errors in
opposite directions, and only a measurement separates them.

**Four things the table says that the counting cannot.**

*The classical basis wins below the crossover by $7.7\times10^{10}$ and the network wins above it by
$5.06$.* Both statements are true and they are not the same size. "The network wins in high
dimensions" is correct and the margin is ten orders of magnitude smaller than the margin it loses by
in low ones.

*The crossover is the grid collapsing, not the network improving.* The fitted growth is $10^{3.0d}$
for the sparse grid against $10^{0.5d}$ for the network. Read the level column for the mechanism: at
a fixed budget the sparse level falls from $16$ to $4$ as $d$ goes from $1$ to $6$, because
$C(\text{level}+d,d)$ grows in both arguments and the budget is fixed.

*Past the crossover neither method is usable.* The worst error there is $1.44$, and a relative error
above $1$ is worse than answering zero everywhere. At four dimensions and above the winner of this
comparison is the less bad of two answers nobody would use. That is what "high dimensional problems
are hard" means, stated as a number.

*Training the hidden layer is worth between $0.02$ and $15$.* In one and two dimensions the frozen
random basis with a least squares output layer reaches $10^{-13}$ and training it with Adam makes it
**worse**, because Adam does not find the optimum a direct solve hands over. From three dimensions
on, where that least squares problem is no longer easy, training earns a factor of up to $14.6$.
**Training helps exactly where solving stops being possible**, which is a much narrower claim than
the one usually made for it and is the one the control supports.

**What this measurement still does not settle.** It uses one manufactured solution, smooth and
separable, which is the case sparse grids are best at. A solution with a sharp front or with genuine
interaction between many coordinates would move the crossover, probably down. The design is here so
that substituting a different `poisson` and rerunning is the whole of the work.

### 4.3 The gap between the two gradients against the solver's order

Section 9 measured the gap falling like $h^{0.998}$ with a fourth order solver for the forward pass
and a first order one for the adjoint. Which order controls it?

```python
import numpy as np
from nalib import sciml


def adjoint_with_order(problem, parameters, steps, order, horizon=1.0):
    """Integrate the adjoint backwards with either a first order or a midpoint rule."""
    d = problem["dimension"]
    theta = np.asarray(parameters, dtype=float)
    w = theta[:d * d].reshape(d, d)
    c = theta[d * d:]
    forward = sciml.integrate(problem, theta, steps, horizon)
    h = forward["step"]
    a = forward["z"] - problem["target"]
    grad = np.zeros(theta.size)
    for index in range(steps - 1, -1, -1):
        z = forward["path"][index]
        later = forward["path"][index + 1]
        if order == 1:
            grad = grad + h * (a @ problem["jacobian_weights"](z, w, c))
            a = a + h * (a @ problem["jacobian_state"](z, w, c))
        else:
            middle = 0.5 * (z + later)
            half = a + 0.5 * h * (a @ problem["jacobian_state"](later, w, c))
            grad = grad + h * (half @ problem["jacobian_weights"](middle, w, c))
            a = a + h * (half @ problem["jacobian_state"](middle, w, c))
    return grad


problem = sciml.flow_problem(3, seed=42)
theta = problem["parameters"]
counts = (16, 32, 64, 128, 256)
print(f"{'steps':>8}{'first order adjoint':>22}{'midpoint adjoint':>20}")
first, second = [], []
for steps in counts:
    reference = sciml.unrolled_gradient(problem, theta, steps)["gradient"]
    scale = float(np.linalg.norm(reference))
    a = float(np.linalg.norm(adjoint_with_order(problem, theta, steps, 1) - reference)) / scale
    b = float(np.linalg.norm(adjoint_with_order(problem, theta, steps, 2) - reference)) / scale
    first.append(a)
    second.append(b)
    print(f"{steps:>8}{a:>22.6e}{b:>20.6e}")

logs = np.log([1.0 / s for s in counts])
print(f"\nfirst order adjoint, fitted order : {np.polyfit(logs, np.log(first), 1)[0]:.4f}")
print(f"midpoint adjoint,    fitted order : {np.polyfit(logs, np.log(second), 1)[0]:.4f}")
```

**The gap is set by the adjoint's own order, not the forward solver's.** The forward pass is
classical Runge-Kutta in both columns, so anything that changes between them is the backward
integration.

**Which answers the practical question.** If the adjoint gradient is wanted to match the discrete one
closely, the fix is a better backward integrator, not a finer mesh. And that is a smaller change than
it looks, because the backward pass reuses the stored forward trajectory and adding a midpoint rule
costs one extra Jacobian evaluation per step.

### 5.1 What a discretization of an ill-posed problem converges to

The uncomfortable answer: **as $n \to \infty$ with fixed noisy data, the discrete solution converges
to nothing useful.** It diverges.

**Why.** The continuous problem $Ax = b$ with $A$ compact has a solution only if $b$ is in the range
of $A$, which is a dense but meagre subset of the data space. Noisy data is not in that range with
probability one. So the continuous problem has **no solution at all**, and the discrete problems are
converging to the non-existence of one: their least squares solutions have norms growing without
bound, which is lesson 98's section 4 measured at $5.8\times10^{7}$ and rising with the noise.

**So the right notion of convergence is a joint one.** The correct statement pairs the discretization
with the regularization and lets both change together. A **regularization method** is a family
$R_\lambda$ with

$$
\lVert R_{\lambda(\delta)}b^\delta - x^\star\rVert \to 0
\qquad\text{as } \delta \to 0 ,
$$

where $b^\delta$ is data with noise level $\delta$ and $\lambda(\delta)$ is a **parameter choice
rule** that must satisfy $\lambda(\delta) \to 0$ and $\delta^2/\lambda(\delta) \to 0$. Both
conditions are needed: the first says the regularization eventually goes away, and the second says it
goes away slower than the noise does.

That second condition is the whole content, and it explains section 4.1's scaling. It rules out
$\lambda = \delta^2$, which is why the measured exponent should be below $2$.

**And the discretization has to be tied in too.** Refining the grid past the point where the
singular values fall below $\delta$ adds only noise, which is lesson 98's section 3: $128$ extra
unknowns and $3$ usable directions. So the discretization level is itself a regularization parameter,
and the practical rule is to choose $n$ so that $\sigma_n \approx \delta$ and no finer.

**Convergence in what norm.** Even with a correct parameter choice rule, the convergence is in the
solution norm and there is no rate without an extra assumption. A **source condition**, $x^\star =
(A^{\mathsf T}A)^{\mu}w$ for some bounded $w$, buys a rate $O(\delta^{2\mu/(2\mu+1)})$, and without
one the convergence can be arbitrarily slow. That is not a gap in the theory; it is a theorem.

### 5.2 When a physics-informed network is the right choice

Section 8 measured a Chebyshev basis beating a trained network by $82000$ on a one-dimensional linear
boundary value problem with a smooth solution. Every word of that description is a condition, and
relaxing any of them changes the answer.

**The features that reverse the conclusion, four of them.**

*High dimension.* A tensor product basis costs $O(m^d)$ and a network costs $O(d)$. Exercise 4.2 puts
the crossover past $d = 6$ on a machine with $64$ GB. Above that there is nothing to compare against,
which is a weaker claim than winning but a decisive one.

*Irregular geometry.* Collocation needs points and a basis; a mesh method needs a mesh. On a domain
that is hard to mesh, a network evaluated at scattered points sidesteps the meshing entirely, and
meshing is often the majority of the human effort in a finite element workflow.

*Unknown terms in the equation.* If a coefficient in the PDE is itself to be learned from data, the
problem is a joint inverse and forward problem. A collocation method with a fixed basis has nothing
to say about the unknown coefficient; a network parametrizes both and fits them together. This is the
strongest case and it is why the method is used in practice.

*Parametric families.* If the same PDE has to be solved for thousands of parameter values, a network
trained once to map parameters to solutions amortizes, where a classical solve does not.

**What measurement would confirm it.** The same three-way comparison as section 8, at $d = 4$ or
$d = 6$, against a sparse-grid or low-rank tensor basis rather than a full tensor product, with the
cost measured in function evaluations rather than in wall-clock time. **The control that section 8
had and most published comparisons lack is the frozen basis**: the same network architecture with its
nonlinear parameters left at their random values and only the output layer fitted. Without it, a win
over a classical method cannot be attributed to the learning rather than to the basis.

Section 8 measured that control at $10.7$: training the basis was worth a factor of $10.7$ over
leaving it random. That is a real number and it is much smaller than the numbers usually quoted.

### 5.3 Discretize or optimize first

**The case for discretize then optimize.** The function you can evaluate is the discrete one. The
training loop computes $L_h(\theta)$, prints it, and decides whether it is falling. If the gradient
used is not $\nabla L_h$, then a line search can fail, the loss can rise on a step the optimizer
believed was a descent direction, and any convergence guarantee built on gradient consistency is
gone. Automatic differentiation gives $\nabla L_h$ exactly and for free.

**The case for optimize then discretize.** Memory: constant against linear in the step count, which
lesson 98's section 9 measured at a ratio of $342$ at $512$ steps and growing. And with an adaptive
solver the discrete function is not even continuous in $\theta$, because the step sequence changes,
so $\nabla L_h$ has jumps and the adjoint's smooth approximation is the better behaved object.

**Which to default to: discretize then optimize, with checkpointing.** Three reasons.

*It gives the gradient of the thing being measured*, so the optimizer and the reported loss agree.

*The memory objection is answerable.* Lesson 97's exercise 2.4: checkpointing takes the memory from
$O(n)$ to $O(\sqrt n)$ at a cost of exactly one extra forward pass, measured at $2.00$ passes at
every gap. That is a much better trade than accepting a different gradient.

*The adjoint's error is not free.* Section 9's gap is $2.2\times10^{-2}$ at $16$ steps, which is
larger than most optimizers' tolerance for gradient error, so getting the adjoint to agree needs a
fine mesh anyway, and a fine mesh is what the memory argument was trying to avoid.

**When to switch.** When the trajectory is genuinely too long to checkpoint, which for a neural ODE
with an adaptive solver and a stiff right-hand side it can be. Then take the adjoint, use a
higher-order backward integrator as exercise 4.3 does, and check it once against a short unrolled
run.

**And note what both routes share.** Neither is more accurate than the forward solve underneath it.
If the ODE is solved to $10^{-3}$, no gradient of that solve means anything at $10^{-6}$, and the
first thing to check when a neural ODE trains badly is the solver tolerance rather than the
differentiation.

---
