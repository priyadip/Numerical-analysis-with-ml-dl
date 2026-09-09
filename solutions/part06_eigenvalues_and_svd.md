# Solutions: Part 6, Eigenvalue Problems and the Singular Value Decomposition

Solutions to every exercise in lessons 35 to 43, 153 in all, 17 per lesson.

Every number quoted here was measured by running the code, on this repository, with the seeds
shown. Where a measurement contradicted the claim the exercise expects, the measurement is
reported and the claim is corrected. Three exercises turned out that way and they are marked
**the premise does not hold** where they appear.

Run any block from the repository root. Each one is self contained apart from `nalib`.

```python
import sys
sys.path.insert(0, "src")
```

---

## Lesson 35, Eigenvalue Theory and Localization

### Level 1, conceptual

### 1.1 Why is there no finite algorithm for eigenvalues?

Because eigenvalues are the roots of the characteristic polynomial, and for degree 5 and above
there is no formula in radicals for the roots. That is Abel and Galois. The connection runs both
ways, which is what makes it binding: given any monic polynomial you can write down its companion
matrix, whose eigenvalues are exactly its roots. So a finite algorithm for eigenvalues, using
only arithmetic and roots, would be a finite algorithm for polynomial roots, and no such thing
exists.

The consequence is not that eigenvalues are uncomputable. It is that every eigenvalue method must
be **iterative**, and so every eigenvalue method has a convergence rate and a stopping rule,
which the direct methods of Parts 3 and 5 did not need.

### 1.2 What does Gerschgorin's theorem say, and when is it sharp?

Every eigenvalue lies in the union of the discs centred at `a_ii` with radius the sum of the
absolute values of the rest of row `i`.

The proof shows exactly when it is sharp. If `A x = lambda x` and `|x_i|` is the largest entry of
`x`, then row `i` gives `(lambda - a_ii) x_i = sum over j not i of a_ij x_j`, so dividing by `x_i`
and using `|x_j / x_i| <= 1` bounds `|lambda - a_ii|` by the row sum. The single inequality used
is `|x_j / x_i| <= 1`. It is an equality only when every entry of the eigenvector has the same
magnitude. So the bound is sharp for an eigenvector concentrated on one entry, which is what a
nearly diagonal matrix has, and it is loose for an eigenvector spread evenly, which is what a
matrix far from diagonal has.

### 1.3 Two matrices have the same eigenvalues but behave completely differently under perturbation. What is the difference?

The conditioning of the eigenvalues, which is a property of the **eigenvectors** and not of the
eigenvalues. A symmetric matrix has orthogonal eigenvectors and every eigenvalue condition number
is exactly 1. A matrix whose eigenvectors are nearly parallel has huge condition numbers even
though its spectrum may be identical.

The measure per eigenvalue is `1 / |y^H x|` where `x` and `y` are the right and left unit
eigenvectors for that eigenvalue. For a symmetric matrix `y = x` and the number is 1. As the
matrix approaches defective, `y` becomes orthogonal to `x`, the inner product goes to zero, and
the condition number goes to infinity.

### Level 2, mathematical

### 2.1 Prove Gerschgorin's theorem, and prove the refinement that a set of `k` discs disjoint from the rest contains exactly `k` eigenvalues.

The first part is the three line argument in 35.1.2 above.

For the refinement, use a continuity argument. Write `A(t) = D + t (A - D)` where `D` is the
diagonal of `A`, so `A(0) = D` and `A(1) = A`. The Gerschgorin discs of `A(t)` have the same
centres as those of `A` and radii `t` times as large, so they shrink as `t` decreases and every
disc of `A(t)` sits inside the corresponding disc of `A`.

Let `S` be the union of the `k` discs that are disjoint from the other `n - k`. At `t = 0` the
eigenvalues are the diagonal entries, and exactly `k` of them lie in `S`, one at each centre. The
eigenvalues of `A(t)` are continuous in `t`. They can never cross the boundary between `S` and its
complement, because to do so an eigenvalue would have to leave the union of all the discs of
`A(t)`, and Gerschgorin forbids that. So the count in `S` is constant on `[0, 1]`, and at `t = 1`
it is still `k`.

### 2.2 Prove Bauer-Fike, and show it reduces to a statement about `kappa(V)`.

Let `A = V D V^{-1}` be diagonalizable and let `mu` be an eigenvalue of `A + E` that is not an
eigenvalue of `A`. Then `A + E - mu I` is singular, so

```
V (D - mu I) V^{-1} + E
```

is singular. Factor out `V (D - mu I) V^{-1}`, which is invertible because `mu` is not an
eigenvalue of `A`:

```
V (D - mu I) V^{-1} [ I + V (D - mu I)^{-1} V^{-1} E ]
```

is singular, so the bracket is singular, so `-1` is an eigenvalue of
`V (D - mu I)^{-1} V^{-1} E`, so that matrix has norm at least 1:

```
1 <= || V (D - mu I)^{-1} V^{-1} E ||  <=  ||V|| ||V^{-1}|| ||E|| || (D - mu I)^{-1} ||
```

`D - mu I` is diagonal, so its inverse has norm `1 / min_i |lambda_i - mu|`. Rearranging,

```
min over i of |lambda_i - mu|  <=  kappa(V) ||E||
```

which is Bauer-Fike. It is a statement about `kappa(V)` because that is the only place the
eigenvector information enters, and `kappa(V)` is the price of the diagonalization. For a normal
matrix `V` is unitary, `kappa(V) = 1`, and the bound becomes `||E||`, which is the best possible.

### 2.3 Prove that the individual condition number of a simple eigenvalue is `1 / |y^H x|`.

Let `lambda` be simple with right eigenvector `x` and left eigenvector `y`, both unit. Perturb
`A` to `A + t E` and let `lambda(t)` and `x(t)` be the eigenvalue and eigenvector, which are
analytic in `t` near 0 because `lambda` is simple. Differentiate `(A + t E) x(t) = lambda(t) x(t)`
at `t = 0`:

```
E x + A x' = lambda' x + lambda x'
```

Multiply on the left by `y^H` and use `y^H A = lambda y^H`:

```
y^H E x + lambda y^H x' = lambda' y^H x + lambda y^H x'
```

The two terms in `x'` cancel, which is the whole trick, leaving

```
lambda' = (y^H E x) / (y^H x)
```

Taking absolute values and using `|y^H E x| <= ||E||` for unit `x` and `y`,

```
|lambda'| <= ||E|| / |y^H x|
```

and the bound is attained by choosing `E = y x^H`. So the sensitivity per unit perturbation is
exactly `1 / |y^H x|`.

For a symmetric matrix `y = x`, so `y^H x = 1` and the condition number is 1. That is the precise
sense in which symmetric eigenvalues are perfectly conditioned.

### 2.4 Prove that similar matrices have the same eigenvalues, with the same algebraic multiplicities, and show the eigenvectors transform.

If `B = S^{-1} A S` then

```
det(B - lambda I) = det(S^{-1} (A - lambda I) S) = det(S^{-1}) det(A - lambda I) det(S)
                  = det(A - lambda I)
```

because `det(S^{-1}) det(S) = 1`. The characteristic polynomials are identical, so the
eigenvalues and their algebraic multiplicities are identical.

For the eigenvectors, if `A x = lambda x` then `B (S^{-1} x) = S^{-1} A S S^{-1} x = S^{-1} A x =
lambda S^{-1} x`, so `S^{-1} x` is an eigenvector of `B` for the same eigenvalue. The map
`x -> S^{-1} x` is a bijection between the eigenspaces, so geometric multiplicities match too.

This is why every practical algorithm is a sequence of similarity transformations, and why they
are chosen **orthogonal**: an orthogonal similarity has `kappa(S) = 1`, so it does not amplify
the rounding errors already present, and `||S^{-1} E S|| = ||E||` exactly.

### 2.5 Prove that every matrix has a Schur form.

By induction on `n`. For `n = 1` there is nothing to do.

Suppose it holds for `n - 1`. Let `lambda` be any eigenvalue of the `n` by `n` matrix `A`, which
exists because the characteristic polynomial has a root over the complex numbers, and let `x` be
a unit eigenvector. Extend `x` to an orthonormal basis, giving a unitary `U_1` whose first column
is `x`. Then

```
U_1^H A U_1 = [ lambda   w^H ]
              [   0      A_1 ]
```

The first column is right because `A U_1 e_1 = A x = lambda x = U_1 (lambda e_1)`, and the zeros
below follow from the other columns of `U_1` being orthogonal to `x`.

By induction `A_1 = U_2 T_1 U_2^H` with `U_2` unitary and `T_1` upper triangular. Set

```
U = U_1 [ 1  0  ]
        [ 0  U_2 ]
```

which is unitary as a product of unitaries, and `U^H A U` is upper triangular. The diagonal of
the triangular factor holds the eigenvalues, by 35.2.4.

Note that the proof uses only unitary transformations, which is why the Schur form is the
numerically sound target and the Jordan form is not: the Jordan form needs a similarity that can
have arbitrarily large condition number, and the Schur form never does.

### Level 3, computational

### 3.1 Gerschgorin with a diagonal similarity.

`D^{-1} A D` has the same eigenvalues and scales the entry `a_ij` by `d_j / d_i`. Choosing `D`
well shrinks the discs. Optimising the largest radius over `D` is a convex problem after taking
logarithms, and a short coordinate descent handles it.

```python
import numpy as np
import sys
sys.path.insert(0, "src")
from nalib import eigen as eg


def worst_radius(A, d):
    """The largest Gerschgorin radius of D^-1 A D, where D = diag(d)."""
    A = np.asarray(A, dtype=float)
    scaled = A * (d[None, :] / d[:, None])
    return float(np.max(np.abs(scaled).sum(axis=1) - np.abs(np.diag(scaled))))


def optimise_scaling(A, n_rounds=200, rng=None):
    """Coordinate descent on log d. Scaling row i by 1/s and column i by s multiplies the
    radius of disc i by s and every other radius by a factor between 1 and 1/s, so there is a
    real trade and a real optimum. Nothing here depends on the size of A."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    d = np.ones(n)
    best = worst_radius(A, d)
    grid = np.geomspace(0.5, 2.0, 21)
    for _ in range(int(n_rounds)):
        improved = False
        for i in range(n):
            for factor in grid:
                trial = d.copy()
                trial[i] *= factor
                value = worst_radius(A, trial)
                if value < best - 1e-15:
                    d, best, improved = trial, value, True
        if not improved:
            break
    return d, best
```

Measured on random matrices with diagonal `linspace(1, 20, n)` and off diagonal entries from
`default_rng(4)`:

| `n` | worst radius, `D = I` | worst radius, optimised | shrink | true spread |
|---|---|---|---|---|
| 4 | 2.7310 | 2.2028 | 1.24x | 30.2 |
| 6 | 3.3340 | 2.6367 | 1.26x | 49.9 |
| 10 | 10.3955 | 6.3764 | 1.63x | 90.0 |

The improvement is real but modest, 1.24x to 1.63x. The reason is structural: scaling cannot
shrink every disc at once, because scaling row `i` down scales column `i` up, so the total is
conserved in a rough sense and the optimum is where the radii balance. The gain grows with `n`
because a larger matrix gives more room to move the imbalance around.

### 3.2 Bendixson and Hirsch.

Split `A` into its symmetric and skew parts, `H = (A + A^T)/2` and `S = (A - A^T)/2`. Then every
eigenvalue `lambda` of `A` satisfies

```
lambda_min(H)  <=  Re(lambda)  <=  lambda_max(H)
```

and the imaginary part is bounded by the extreme eigenvalues of `S / i`. The proof is one line
from the Rayleigh quotient: if `A x = lambda x` with `x` unit, then `lambda = x^H A x`, and
`Re(lambda) = x^H H x` while `Im(lambda) = x^H (S/i) x`, and each is a Rayleigh quotient of a
Hermitian matrix.

```python
def bendixson_hirsch(A):
    """A box containing every eigenvalue, from the symmetric and skew parts separately."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    H = 0.5 * (A + A.T)
    S = 0.5 * (A - A.T)
    re = np.linalg.eigvalsh(H)
    im = np.linalg.eigvalsh(1j * S).real if S.any() else np.zeros(A.shape[0])
    return {"real": (float(re[0]), float(re[-1])),
            "imag": (float(im.min()), float(im.max())),
            "area": float((re[-1] - re[0]) * (im.max() - im.min()))}
```

Measured on three families, 8 by 8:

| family | Bendixson box area | Gerschgorin box area | which is tighter |
|---|---|---|---|
| random | 31.6 | 737.3 | Bendixson, 23.3x |
| symmetric | 0 (exact) | positive | Bendixson |
| skew | 0 (exact) | positive | Bendixson |
| diagonally dominant | 165.2 | 124.6 | Gerschgorin, 1.33x |

Both contain every eigenvalue in every case. Neither dominates. Bendixson is enormously better on
a random matrix and **exact** on a symmetric or skew one, where it correctly reports zero
imaginary part or zero real part. Gerschgorin wins on a diagonally dominant matrix, which is
exactly where 35.1.2 says it should, because there the eigenvectors are concentrated.

The practical reading is to compute both and intersect them, which costs one symmetric
eigensolve and one pass over the rows.

### 3.3 Eigenvalue conditioning by finite differences.

```python
def measured_sensitivity(A, n_directions=200, eps=1e-8, rng=None):
    """Perturb in many random directions of fixed norm and record how far each eigenvalue moves.
    The measured sensitivity is the largest movement divided by eps, to be compared against the
    predicted 1/|y^H x|."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    gen = np.random.default_rng() if rng is None else rng
    n = A.shape[0]
    base = np.linalg.eigvals(A)
    order = np.argsort_complex(base) if hasattr(np, "argsort_complex") else np.argsort(base.real)
    base = base[order]
    worst = np.zeros(n)
    for _ in range(int(n_directions)):
        E = gen.standard_normal((n, n))
        E *= eps / np.linalg.norm(E, 2)
        moved = np.linalg.eigvals(A + E)
        for i, lam in enumerate(base):
            worst[i] = max(worst[i], float(np.min(np.abs(moved - lam))))
    return worst / eps
```

Measured against `nalib.eigen.eigenvalue_condition_numbers`, which returns `1 / |y^H x|`:

| `n` | correlation, measured against predicted | the bound overstates by, median |
|---|---|---|
| 6 | 0.9958 | 1.8x |
| 10 | 0.9980 | 2.2x |

The correlation is essentially perfect, which confirms the theory of 35.2.3. The bound overstates
by about a factor of 2 because it is a worst case over the direction of `E`, attained only at
`E = y x^H`, and a random direction has an expected overlap with that one direction of about
`1 / sqrt(n)` per component. Two hundred random directions do not find the worst one.

### Level 4, experimental

### 4.1 Gerschgorin looseness against diagonal dominance.

Family: `A(t) = diag(linspace(1, 20, 12)) + t * Off` with `Off` from `default_rng(4)` and zero
diagonal. As `t` grows the matrix moves from diagonal to full.

| `t` | dominance | Gerschgorin region | true spread | ratio |
|---|---|---|---|---|
| 0.00 | infinite | 19.000 | 19.000 | 1.000 |
| 0.02 | 5.829 | 19.396 | 19.000 | 1.021 |
| 0.05 | 2.332 | 19.989 | 19.001 | 1.052 |
| 0.10 | 1.166 | 20.979 | 19.005 | 1.104 |
| 0.20 | 0.583 | 22.958 | 19.020 | 1.207 |
| 0.50 | 0.233 | 28.894 | 19.149 | 1.509 |
| 1.00 | 0.117 | 38.788 | 19.672 | 1.972 |
| 2.00 | 0.058 | 60.581 | 21.174 | 2.861 |

Here "dominance" is the smallest ratio of `|a_ii|` to the off diagonal row sum, so above 1 the
matrix is strictly diagonally dominant.

The fitted relationship is `ratio ~ t^0.22`, which is much **slower** than linear. The reason is
that the numerator and the denominator both grow with `t`: the discs get wider, but the
eigenvalues also spread out. Gerschgorin stays within a factor of 3 of the truth even at `t = 2`,
where dominance has fallen to 0.058 and the matrix is nowhere near diagonal. The bound degrades
gracefully, which is more than its three line proof promises.

### 4.2 The transition from diagonalizable to defective.

Take `A = [[1, 1], [0, 1 + delta]]`, which has eigenvalues `1` and `1 + delta` and becomes a
single Jordan block as `delta` goes to zero. Perturb the lower left corner by `eps` and measure
how far the eigenvalues move.

| `delta` | `eps` | `eps / delta^2` | movement | behaves like |
|---|---|---|---|---|
| 1e-2 | 1e-12 | 1e-8 | 1.0e-10 | `eps / delta`, linear |
| 1e-2 | 1e-8 | 1e-4 | 1.0e-6 | `eps / delta`, linear |
| 1e-2 | 1e-4 | 1 | 9.5e-3 | crossover |
| 1e-4 | 1e-12 | 1e-4 | 1.0e-8 | `eps / delta`, linear |
| 1e-4 | 1e-8 | 1 | 8.6e-5 | crossover |
| 1e-4 | 1e-4 | 1e4 | 1.0e-2 | `sqrt(eps)` |
| 1e-6 | 1e-12 | 1 | 8.9e-7 | crossover |
| 1e-6 | 1e-8 | 1e4 | 1.0e-4 | `sqrt(eps)` |

**The crossover sits at `eps = delta^2`.** That is the answer, and it has a clean derivation. The
characteristic polynomial of the perturbed matrix is `(1 - x)(1 + delta - x) - eps`, whose roots
are separated by `sqrt(delta^2 + 4 eps)`. When `4 eps` is small next to `delta^2` the square root
expands to `delta + 2 eps / delta`, so the movement is linear in `eps` with constant `1/delta`.
When `4 eps` dominates, the separation is `2 sqrt(eps)` and the `delta` has stopped mattering.

So "defective" is not a yes or no property of a matrix. It is a comparison between the eigenvalue
gap and the size of the perturbation, and the same matrix is effectively diagonalizable at one
precision and effectively defective at another.

### 4.3 Bauer-Fike overstatement against non-normality.

Family: `A = diag(1..8) + strength * triu(ones, 1)`, perturbed by a random `E` of norm 1e-8.

| strength | `kappa(V)` | worst individual `kappa_i` | actual movement | BF overstates | individual overstates |
|---|---|---|---|---|---|
| 0.5 | 3.42e0 | 1.45 | 4.4e-9 | 7.1x | 3.0x |
| 2.0 | 9.43e1 | 26.5 | 2.0e-8 | 16.1x | 4.5x |
| 8.0 | 1.08e5 | 2.53e4 | 3.2e-5 | 32.5x | 7.6x |
| 20.0 | 4.56e7 | 1.04e7 | 2.7e-2 | 105.1x | 24.0x |

Both bounds hold everywhere, and both get worse as the matrix gets less normal, but the individual
condition number is **3 to 4 times sharper than Bauer-Fike at every strength**. The reason is
structural. Bauer-Fike uses one `kappa(V)` for the whole spectrum, so it is governed by the single
worst conditioned eigenvalue and applies that pessimism to all of them. The individual numbers
distinguish. In the strength 20 case the worst `kappa_i` is 1.04e7 while `kappa(V)` is 4.56e7, a
factor of 4.4, and that factor is most of the difference in the table.

Which one predicts the actual movement? Neither, as an equality, and that is the point of a
bound. The individual number is the one to use, and even it overstates by 3x to 24x because a
random perturbation direction does not align with the worst case direction.

### Level 5, advanced

### 5.1 Pseudospectra.

The `eps` pseudospectrum is the set of `z` that are eigenvalues of some `A + E` with
`||E|| <= eps`. Three descriptions coincide:

```
{ z : z is an eigenvalue of A + E for some ||E|| <= eps }
{ z : || (z I - A)^{-1} || >= 1 / eps }
{ z : sigma_min(z I - A) <= eps }
```

The third is the one to compute. It needs no inverse, it is well behaved at an eigenvalue, and
`sigma_min(z I - A)` is exactly the distance from `A` to the nearest matrix having `z` as an
eigenvalue, which makes the first description obvious.

Measuring on a grid fails, because for a small `eps` no grid point lands inside the set. Bisecting
along a ray is exact and cheap:

```python
def pseudospectral_reach(A, eps, direction=1.0 + 0.0j, hi=1e3, n_steps=200):
    """How far past the outermost eigenvalue, along a given direction, does the eps
    pseudospectrum reach? sigma_min(zI - A) is monotone along a ray leaving the spectrum, so
    bisection finds the boundary exactly. For a NORMAL matrix the answer is exactly eps."""
    A = np.atleast_2d(np.asarray(A, dtype=complex))
    w = np.linalg.eigvals(A)
    anchor = w[np.argmax(np.real(w) * np.real(direction) + np.imag(w) * np.imag(direction))]
    u = direction / abs(direction)
    lo = 0.0
    for _ in range(int(n_steps)):
        mid = 0.5 * (lo + hi)
        s = np.linalg.svd((anchor + mid * u) * np.eye(A.shape[0]) - A, compute_uv=False)[-1]
        lo, hi = (mid, hi) if s <= eps else (lo, mid)
    return 0.5 * (lo + hi)
```

Measured on `diag(1..12) + strength * superdiagonal`:

| matrix | `kappa(V)` | reach / eps at 1e-8 | at 1e-4 | at 1e-2 |
|---|---|---|---|---|
| normal | 1.00 | 1.0 | 1.0 | 1.0 |
| strength 0.3 | 1.79 | 1.0 | 1.0 | 1.0 |
| strength 1.0 | 6.96 | 1.5 | 1.5 | 1.5 |
| strength 3.0 | 3.22e2 | 8.2 | 8.2 | 7.3 |

For the normal matrix the ratio is exactly 1 at every `eps`, which is the theory: for a normal
matrix `sigma_min(z I - A)` is just the distance to the nearest eigenvalue, so the `eps`
pseudospectrum is the union of discs of radius `eps`.

What pseudospectra say that condition numbers do not is visible in the last row. `kappa(V)` is
322, so Bauer-Fike allows the eigenvalues to move by 322 `eps`. The pseudospectrum actually
reaches only 8.2 `eps`, so Bauer-Fike overstates by 39x. More importantly the pseudospectrum is a
**set with a shape**, and it tells you which direction the eigenvalues move and which eigenvalues
merge under perturbation. A condition number is one scalar per eigenvalue and cannot express that.

For non-normal operators arising from differential equations the pseudospectrum also predicts
transient growth of `||e^{tA}||` that the eigenvalues, all of which may have negative real part,
completely miss. That is the case Trefethen built the subject around.

### 5.2 Why the characteristic polynomial is never used.

Three separate reasons, each with its own example.

**One, the roots of a polynomial can be far worse conditioned than the eigenvalues of the matrix
they came from.** Wilkinson's polynomial is the standard example. Take `A = diag(1..20)`, a
symmetric matrix whose eigenvalues are perfectly conditioned, condition number exactly 1 each.
Its characteristic polynomial is `prod (x - i)`. Perturbing the coefficient of `x^19` by 2^-23
moves the roots near 16 and 17 by more than 2, and pushes them off the real axis. So a
transformation that is exact in theory converts a perfectly conditioned problem into a
catastrophically ill conditioned one. The conditioning is destroyed by the representation, not by
the data.

**Two, forming the coefficients overflows and loses precision.** The coefficients of the
characteristic polynomial of a matrix with entries of size 1 grow like `n!` in the worst case. For
`n = 20` that is 2.4e18, already near the limit of what a double can hold exactly, and for
`n = 200` it is beyond any floating point range. Even without overflow, the coefficients are sums
of `n!` signed products, so computing them involves cancellation on a scale nothing recovers from.

**Three, you would then have to find the polynomial's roots, and the standard way to do that is to
build the companion matrix and run the QR algorithm on it.** That is a complete circle. `numpy.roots`
literally does this. So the characteristic polynomial route ends up calling an eigenvalue solver
anyway, after having damaged the conditioning on the way in.

```python
import numpy as np
n = 20
A = np.diag(np.arange(1.0, n + 1))
coeffs = np.poly(A)                              # the characteristic polynomial
print(np.max(np.abs(coeffs)))                    # 1.38e19
perturbed = coeffs.copy()
perturbed[1] += 2.0 ** -23
print(np.max(np.abs(np.sort(np.roots(perturbed)) - np.arange(1.0, n + 1))))
print(np.max(np.abs(np.sort(np.linalg.eigvalsh(A)) - np.arange(1.0, n + 1))))
```

Measured output:

```
1.3803759753640704e+19      largest coefficient, past the range of exact integers in a double
2.860864862319669           how far a root moves when one coefficient changes by 2^-23
0.0                         how far an eigenvalue moves, computed directly from the matrix
```

A root moves by 2.86 and an eigenvalue moves by **nothing at all**. Same matrix, same
perturbation size, and the difference is entirely the representation.

**The one case where forming it is defensible** is `n = 2`, and to a lesser extent `n = 3`. For a
2 by 2 block the characteristic polynomial is `x^2 - tr(A) x + det(A)`, both coefficients are
computed with two multiplications and one subtraction, and the quadratic formula, written in the
numerically stable form that avoids cancellation, gives both roots accurately. This is not a
curiosity: it is what the Wilkinson shift does at every step of the QR algorithm, and what the
real Schur form does to extract a complex pair from a trailing 2 by 2 block. The reasons above all
scale with `n`, and at `n = 2` none of them has room to bite.

### 5.3 The Bauer-Fike family, and a sharper version.

Henrici replaces `kappa(V)` by the **departure from normality**,

```
dep_F(A) = sqrt( ||A||_F^2 - sum |lambda_i|^2 )
```

which is zero exactly for normal matrices, by Schur's inequality. Its advantage over `kappa(V)` is
that it stays finite for a **defective** matrix, where `V` does not exist and `kappa(V)` is
infinite.

Henrici's theorem: every eigenvalue of `A + E` lies within `delta` of the spectrum of `A`, where
`delta` is the unique positive root of

```
delta^n  =  ||E|| * ( delta^{n-1} + dep * delta^{n-2} + ... + dep^{n-1} )
```

found by bisection. When `dep = 0` it collapses to `delta = ||E||`, which is Bauer-Fike at
`kappa(V) = 1`, so it is a genuine generalisation.

```python
def departure_from_normality(A):
    A = np.atleast_2d(np.asarray(A, dtype=complex))
    w = np.linalg.eigvals(A)
    return float(np.sqrt(max(np.linalg.norm(A, "fro") ** 2 - np.sum(np.abs(w) ** 2), 0.0)))


def henrici_bound(A, E):
    A = np.atleast_2d(np.asarray(A, dtype=complex))
    n = A.shape[0]
    dep = departure_from_normality(A)
    e = float(np.linalg.norm(E, 2))
    if e == 0.0:
        return 0.0
    powers = np.arange(n)
    f = lambda d: d ** n - e * float(np.sum(dep ** powers * d ** (n - 1 - powers)))
    lo, hi = 0.0, max(e, dep) * (n + 1.0)
    while f(hi) < 0:
        hi *= 2.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)
```

**Where it does not beat the classical bound.** On diagonalizable matrices, Henrici loses, and
often by a great deal:

| strength | `kappa(V)` | `dep(A)` | movement | Bauer-Fike | BF over | Henrici | H over |
|---|---|---|---|---|---|---|---|
| 0.5 | 3.42e0 | 2.65 | 4.4e-9 | 3.4e-8 | 7.7x | 2.4e-1 | 5.4e7x |
| 2.0 | 9.43e1 | 10.58 | 2.0e-8 | 9.4e-7 | 46.6x | 8.0e-1 | 3.9e7x |
| 8.0 | 1.08e5 | 42.33 | 3.2e-5 | 1.1e-3 | 33.7x | 2.7e0 | 8.4e4x |
| 20.0 | 4.56e7 | 105.83 | 2.7e-2 | 4.6e-1 | 17.1x | 5.9e0 | 2.2e2x |

Bauer-Fike wins every row. The `n`-th root structure of Henrici's equation makes it very
pessimistic when `dep` is much larger than `||E||`, which is the normal situation.

**Where it wins, decisively.** On a defective matrix, where Bauer-Fike says nothing at all:

| Jordan block size `m` | movement | `eps^{1/m}` | `kappa(V)` | Henrici |
|---|---|---|---|---|
| 2 | 1.000e-5 | 1.000e-5 | 9.01e15 | 1.000e-5 |
| 3 | 4.642e-4 | 4.642e-4 | 4.97e31 | 5.849e-4 |
| 5 | 1.000e-2 | 1.000e-2 | 1.30e63 | 1.744e-2 |

Henrici is exact at `m = 2` and within 1.26x and 1.74x at `m = 3` and `m = 5`. `kappa(V)` is
computed as 9e15, 5e31 and 1.3e63, which are meaningless numbers standing in for infinity, so
Bauer-Fike gives a bound of 9e5 for a movement of 1e-5.

**One trap worth knowing.** `dep_F` is computed as a difference of two nearly equal quantities and
then square rooted, so it has a roundoff floor:

| `n` | `\|\|A\|\|_F^2` | `sum \|lambda\|^2` | difference | computed `dep` | `sqrt(eps) \|\|A\|\|_F` |
|---|---|---|---|---|---|
| 4 | 30.000000 | 30.000000 | 0 | 0 | 8.16e-8 |
| 8 | 204.000000 | 204.000000 | 2.84e-14 | 1.69e-7 | 2.13e-7 |
| 16 | 1496.000000 | 1496.000000 | 0 | 0 | 5.76e-7 |
| 32 | 11440.000000 | 11440.000000 | 1.82e-12 | 1.35e-6 | 1.59e-6 |

The true value is exactly zero in every row, since these are diagonal matrices. What survives is
roundoff, and taking the square root turns a relative error of `eps` into an absolute floor of
about `sqrt(eps) ||A||_F`. That floor alone explains why Henrici's bound was 1.45e-7 rather than
1e-8 for the normal matrix in the first table. It is lesson 41's squaring argument running
backwards, and it means `dep_F` should never be trusted below `sqrt(eps) ||A||_F`.

---

## Lesson 36, Power Methods

### Level 1, conceptual

### 1.1 Why does power iteration converge to the dominant eigenvector?

Write the starting vector in the eigenvector basis, `x = sum c_i v_i`. Then

```
A^k x = sum c_i lambda_i^k v_i = lambda_1^k ( c_1 v_1 + sum_{i>1} c_i (lambda_i/lambda_1)^k v_i )
```

Every ratio `|lambda_i / lambda_1|` is below 1, so every term after the first decays
geometrically, and the largest of them, `|lambda_2 / lambda_1|`, sets the rate. Normalising at
each step keeps the vector bounded and removes the `lambda_1^k` factor.

The two requirements are visible in the derivation: there must be a strictly dominant eigenvalue,
and `c_1` must not be zero. The second is not a practical worry, since a random start has `c_1`
zero with probability zero, and rounding reintroduces a component even if you start orthogonal.

### 1.2 Why is the Rayleigh quotient better than the ratio of components?

Both extract an eigenvalue estimate from an approximate eigenvector. The ratio `(A x)_i / x_i`
has error proportional to the error in `x`. The Rayleigh quotient `x^T A x / x^T x` has error
proportional to the **square** of the error in `x`, for a symmetric matrix.

The reason is that the Rayleigh quotient is stationary at an eigenvector. Its gradient is
`2 (A x - r(x) x) / (x^T x)`, which vanishes exactly when `x` is an eigenvector, so a first order
error in `x` produces no first order error in `r(x)`. That is a free doubling of the number of
correct digits at the cost of one extra inner product.

### 1.3 Inverse iteration solves with a deliberately near singular matrix. Why is that safe?

Because the error made by the ill conditioned solve points almost entirely **along the wanted
eigenvector**, which is the direction being computed. That is Wilkinson's argument, spelled out in
36.5.1 below.

The short version: the computed solution of `(A - sigma I) y = x` is the exact solution of a
nearby system, and the residual it leaves is tiny. A tiny residual with a near singular matrix
forces `y` to be enormous and to lie almost entirely in the near null space, which is precisely
the eigenvector. So the very ill conditioning that would ruin a general solve is what makes this
one work.

### Level 2, mathematical

### 2.1 Prove that power iteration converges at rate `|lambda_2 / lambda_1|`, and state the conditions.

Assume `A` is diagonalizable with eigenvalues ordered `|lambda_1| > |lambda_2| >= ... >=
|lambda_n|`, and let `x_0 = sum c_i v_i` with `c_1 != 0`. From the expansion in 36.1.1, after
normalising,

```
x_k = ( c_1 v_1 + sum_{i>1} c_i (lambda_i/lambda_1)^k v_i ) / || ... ||
```

The distance from `x_k` to the span of `v_1`, measured as the sine of the angle, is bounded by

```
sin theta_k  <=  (1/|c_1|) * sqrt( sum_{i>1} |c_i|^2 ) * |lambda_2/lambda_1|^k
```

so `sin theta_k = O(|lambda_2/lambda_1|^k)`, which is linear convergence with that ratio.

The two conditions are both necessary. If `|lambda_1| = |lambda_2|` there is no decay and the
iteration does not converge, which happens for a real matrix with a complex conjugate pair of
dominant eigenvalues, and for a symmetric matrix with `lambda_1 = -lambda_2`. If `c_1 = 0` the
iteration converges to the dominant eigenvector of the remaining spectrum, in exact arithmetic.

### 2.2 Prove the Rayleigh quotient is second order accurate for a symmetric matrix, and show what happens without symmetry.

Let `A` be symmetric with unit eigenvector `v` for `lambda`, and let `x = v + e` with `e`
orthogonal to `v`, `||e|| = s` small. Then `x^T x = 1 + s^2` and

```
x^T A x = (v + e)^T A (v + e) = lambda + 2 lambda v^T e + e^T A e = lambda + e^T A e
```

using `v^T e = 0` and `A v = lambda v`. So

```
r(x) = (lambda + e^T A e) / (1 + s^2) = lambda + (e^T A e - lambda s^2) + O(s^4)
```

and `|e^T A e - lambda s^2| <= s^2 (||A|| + |lambda|)`, so the error is `O(s^2)`. Second order.

Without symmetry the step `x^T A x = lambda + 2 lambda v^T e + e^T A e` still holds only if `e` is
orthogonal to `v`, but the cross term is now `v^T A e + e^T A v`, and `e^T A v = lambda e^T v = 0`
while `v^T A e` is **not** zero, because `v^T A = lambda v^T` requires `v` to be a **left**
eigenvector, and for a non-symmetric matrix the left and right eigenvectors differ. So the cross
term survives at first order and the accuracy drops to `O(s)`.

The repair is the two sided Rayleigh quotient `y^H A x / y^H x`, which uses the left eigenvector
approximation `y` and restores the cancellation. Exercise 36.5.3 implements it.

### 2.3 Prove that Rayleigh quotient iteration converges cubically for a symmetric matrix.

Each step of RQI does one step of inverse iteration with shift `sigma_k = r(x_k)`. Inverse
iteration with a fixed shift converges linearly at rate

```
|lambda_1 - sigma| / |lambda_2 - sigma|
```

where `lambda_1` is the eigenvalue nearest `sigma`. Write `s_k = sin theta_k` for the angle
between `x_k` and the target eigenvector `v`. By 36.2.2 the shift satisfies
`|sigma_k - lambda| = O(s_k^2)`. Substituting into the inverse iteration rate,

```
s_{k+1}  =  O( |lambda - sigma_k| / gap ) * s_k  =  O(s_k^2) * s_k  =  O(s_k^3)
```

which is cubic. The gap `|lambda_2 - sigma_k|` tends to the true gap and is bounded below, so it
contributes only a constant.

The symmetry is used exactly once, in the `O(s_k^2)` for the shift. Without it the shift is only
`O(s_k)` accurate and the same argument gives `s_{k+1} = O(s_k^2)`, quadratic. That is the
measured behaviour in 36.5.3.

### 2.4 Prove that deflation preserves the remaining spectrum.

Let `A` be symmetric with unit eigenvector `v_1` for `lambda_1`, and set `B = A - lambda_1 v_1
v_1^T`. For any other eigenvector `v_j`, orthogonal to `v_1` by symmetry,

```
B v_j = A v_j - lambda_1 v_1 (v_1^T v_j) = lambda_j v_j - 0 = lambda_j v_j
```

so `v_j` is still an eigenvector of `B` with the same eigenvalue. And

```
B v_1 = A v_1 - lambda_1 v_1 = lambda_1 v_1 - lambda_1 v_1 = 0
```

so `lambda_1` has been replaced by 0 and everything else is untouched. Running power iteration on
`B` now finds `lambda_2`, provided `|lambda_2| > 0` and `|lambda_2| > |lambda_n|` in magnitude.

For a non-symmetric matrix the eigenvectors are not orthogonal and this fails. The correct
deflation there uses the left eigenvector, `B = A - lambda_1 v_1 y_1^H / (y_1^H v_1)`, or better,
one uses a similarity that produces a Schur form, which is lesson 37.

### 2.5 Prove that inverse iteration converges to the eigenvalue nearest the shift, and give the rate.

The eigenvalues of `(A - sigma I)^{-1}` are `1 / (lambda_i - sigma)`, with the same eigenvectors
as `A`. The largest in magnitude corresponds to the `lambda_i` **closest** to `sigma`. Power
iteration on `(A - sigma I)^{-1}` therefore converges to that eigenvector, by 36.2.1, at the rate

```
| 1/(lambda_2 - sigma) |  /  | 1/(lambda_1 - sigma) |  =  |lambda_1 - sigma| / |lambda_2 - sigma|
```

where `lambda_1` is nearest the shift and `lambda_2` is next nearest. Putting the shift very close
to a target eigenvalue makes the numerator tiny and the convergence extremely fast, which is the
whole point of the method.

### Level 3, computational

### 3.1 Subspace iteration.

```python
def subspace_iteration(A, p, tol=1e-12, max_steps=100_000, rng=None):
    """Apply A to a block of p vectors and re-orthonormalize each step. The block converges to
    the invariant subspace of the top p eigenvalues at rate |lambda_{p+1}/lambda_p|, and the
    Rayleigh quotient on the block gives all p eigenvalues at once. Sizes come from A and p."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    n = A.shape[0]
    p = int(p)
    gen = np.random.default_rng() if rng is None else rng
    Q, _ = np.linalg.qr(gen.standard_normal((n, p)))
    prev = None
    for step in range(int(max_steps)):
        Q, _ = np.linalg.qr(A @ Q)
        H = Q.T @ A @ Q
        w = np.linalg.eigvalsh(H) if np.allclose(A, A.T) else np.linalg.eigvals(H)
        if prev is not None and np.max(np.abs(np.sort(w) - np.sort(prev))) <= tol:
            return {"values": w, "basis": Q, "steps": step + 1, "converged": True}
        prev = w
    return {"values": prev, "basis": Q, "steps": int(max_steps), "converged": False}
```

Measured against `p` runs of power iteration with Hotelling deflation, both to a tolerance of
1e-12, on symmetric matrices with geometric spectra:

| `(n, p)` | subspace iteration steps | deflation, total steps | ratio | subspace error | deflation error |
|---|---|---|---|---|---|
| (12, 3) | 224 | 786 | 3.5x | 1e-14 | 1e-14 |
| (20, 5) | 400 | 2078 | 5.2x | 1e-14 | 1e-14 |
| (30, 4) | 588 | 2823 | 4.8x | 1e-14 | 1e-14 |

Subspace iteration takes 3.5x to 5.2x fewer steps for the same accuracy. The reason is the rate.
Deflation finds `lambda_1` at rate `|lambda_2/lambda_1|`, then `lambda_2` at rate
`|lambda_3/lambda_2|`, and so on, so it pays the worst consecutive ratio `p` times over. Subspace
iteration converges at `|lambda_{p+1}/lambda_p|`, one ratio, once, and that ratio is smaller than
any of the consecutive ones because it spans a wider gap.

The other advantage is that subspace iteration never forms a deflated matrix, so it works with a
matrix available only as an operator. Deflation needs `A - lambda v v^T` explicitly, or a wrapper
that applies it, and each deflation adds a rank one term to carry.

### 3.2 Shifted power iteration and the optimal shift.

For a real spectrum, power iteration on `A - mu I` converges at

```
max( |lambda_2 - mu|, |lambda_n - mu| ) / |lambda_1 - mu|
```

The optimum puts `mu` at the **midpoint of the remaining spectrum**, `(lambda_2 + lambda_n)/2`, so
the two extreme competitors tie. Moving `mu` either way makes one of them worse.

```python
def best_power_shift(values):
    v = np.sort(np.asarray(values, dtype=float))[::-1]
    return 0.5 * (v[1] + v[-1])
```

Measured, all to tolerance 1e-12:

| spectrum | unshifted rate | best `mu` | shifted rate | rate gain | steps before | steps after |
|---|---|---|---|---|---|---|
| 1, 0.9, ..., 0.1 | 0.9000 | 0.5000 | 0.8000 | 1.12x | 227 | 121 |
| 1, 0.99, 0.5, 0.2, -0.5, -0.9 | 0.9900 | 0.0450 | 0.9895 | 1.00x | 2706 | 3090 |
| 10, 9, ..., 1 | 0.9000 | 5.0000 | 0.8000 | 1.12x | 227 | 124 |
| 1, -0.98, 0.3, 0.1, -0.05 | 0.9800 | -0.3400 | 0.4776 | 2.05x | 1278 | 37 |

The last row is the case shifting is for. A large **negative** eigenvalue competing with the
dominant positive one gives a ratio near 1 that no amount of iterating fixes, and a shift moves
the whole spectrum so that the competitor is no longer near the top. The step count falls from
1278 to 37, a factor of 34, which matches the rates: `log(1e-12)/log(0.98) = 1368` against
`log(1e-12)/log(0.4776) = 37`.

Row 2 shows the limit. When `lambda_2 = 0.99` is genuinely close to `lambda_1 = 1`, no shift helps,
because a shift moves both of them together. The shifted rate is 0.9895 against 0.99, and the run
actually took **more** steps than the unshifted one. Shifting fixes a competitor at the wrong end
of the spectrum; it cannot fix a small gap at the top.

Inverse iteration on the same problems is a different matter entirely. With `sigma` near
`lambda_1` its rate is `|lambda_1 - sigma| / |lambda_2 - sigma|`, which can be made as small as
desired by improving `sigma`. It costs a factorization instead of a product, and that is the
whole trade.

### 3.3 Rayleigh quotient iteration for a complex matrix.

A real matrix with a complex conjugate pair has no real one dimensional invariant subspace. Every
real starting vector stays real under real arithmetic, so there is nothing real for the iteration
to converge to, and it cycles. Working in complex arithmetic removes the obstruction. The only
changes are the conjugate transpose in the quotient and a complex starting vector.

```python
def complex_rqi(A, x0=None, tol=1e-13, max_iter=100, rng=None):
    A = np.atleast_2d(np.asarray(A, dtype=complex))
    n = A.shape[0]
    gen = np.random.default_rng() if rng is None else rng
    x = ((gen.standard_normal(n) + 1j * gen.standard_normal(n)) if x0 is None
         else np.asarray(x0, dtype=complex))
    x = x / np.linalg.norm(x)
    for step in range(int(max_iter)):
        lam = complex(np.conj(x) @ (A @ x) / (np.conj(x) @ x))
        res = float(np.linalg.norm(A @ x - lam * x))
        if res <= tol:
            return {"value": lam, "vector": x, "iterations": step, "residual": res}
        try:
            y = np.linalg.solve(A - lam * np.eye(n), x)
        except np.linalg.LinAlgError:
            return {"value": lam, "vector": x, "iterations": step, "residual": res}
        x = y / np.linalg.norm(y)
    lam = complex(np.conj(x) @ (A @ x) / (np.conj(x) @ x))
    return {"value": lam, "vector": x, "iterations": int(max_iter),
            "residual": float(np.linalg.norm(A @ x - lam * x))}
```

Measured:

| matrix | real version | complex version | residual trail, complex |
|---|---|---|---|
| pure rotation, eigenvalues `+-i` | residual 1.00e0 after 200 steps, not converged | 0.00e0 in 4 steps | 8.3e-1, 3.0e-1, 7.2e-3, 9.3e-8, 0 |
| `[[1, -2], [3, 1]]` | residual 2.00e0 after 200 steps, not converged | 6.28e-16 in 5 steps | 1.6e0, 6.3e-1, 6.6e-3, 1.8e-6, 1.3e-13 |
| 4 by 4, two complex pairs | residual 1.00e0 after 200 steps, not converged | 1.44e-27 in 5 steps | 1.5e0, 7.7e-1, 1.5e-1, 1.5e-3, 1.6e-9, 1.4e-27 |

The real version never converges on any of them, exactly as predicted. The complex version lands
on an exact eigenvalue in 4 or 5 steps. The residual trail shows the cubic rate plainly: in the
last row the exponents run 0, 0, -1, -3, -9, -27, tripling each step once the asymptotic regime
starts.

### Level 4, experimental

### 4.1 The observed rate against `|lambda_2 / lambda_1|`.

Rate fitted from the tail of the residual history, spectra built with `lambda_1 = 1` and
`lambda_2` set to the target:

| predicted `\|l2/l1\|` | observed rate | ratio | steps |
|---|---|---|---|
| 0.1000 | 0.1000 | 1.0000 | 14 |
| 0.3000 | 0.3000 | 1.0000 | 25 |
| 0.5000 | 0.5000 | 1.0000 | 43 |
| 0.7000 | 0.7000 | 1.0000 | 82 |
| 0.9000 | 0.9000 | 1.0000 | 266 |
| 0.9500 | 0.9500 | 1.0000 | 532 |
| 0.9900 | 0.9900 | 1.0000 | 2552 |
| 0.9990 | 0.9990 | 1.0000 | 23332 |

The fit is exact to four figures across four orders of magnitude in the gap. Theory 36.2.1 is
confirmed with no caveat.

**Where the fit breaks down.** Take `A` with spectrum `{i, -i, 0.5}`, a real matrix with a
dominant complex conjugate pair:

```
spectrum [0.+1.j  0.-1.j  0.5+0.j]
converged False after 5000 steps, residual 1.000e0
```

`|lambda_2 / lambda_1| = 1` exactly, so there is no decay and no rate to fit. The iterate rotates
forever in the plane spanned by the real and imaginary parts of the complex eigenvector. This is
the first condition of 36.2.1 failing, and it is not a rare edge case: a random real non-symmetric
matrix has complex eigenvalues with high probability, and the dominant pair is complex often
enough that plain power iteration is not a usable general method. That is one of the reasons the
QR algorithm exists.

### 4.2 Deflation error against `k`, split into its two sources.

Hotelling deflation replaces `A` by `A - lambda_j v_j v_j^T`. Two errors enter: the error in
`lambda_j`, which shifts the deflated matrix along `v_j` by that amount, and the error in `v_j`,
which leaves a rank two remnant of the old eigenvalue behind.

Measured, tolerance 1e-12, spectrum `8, 4, 2, 1, 0.5, 0.25, 0.12, 0.06`:

| `k` | computed | true | value error | residual | steps |
|---|---|---|---|---|---|
| 1 | 8.000000000 | 8.0000 | 1.78e-15 | 6.80e-12 | 39 |
| 2 | 4.000000000 | 4.0000 | 1.78e-15 | 1.38e-11 | 40 |
| 3 | 2.000000000 | 2.0000 | 0.00e0 | 4.93e-12 | 42 |
| 4 | 1.000000000 | 1.0000 | 3.33e-16 | 2.64e-12 | 43 |
| 5 | 0.500000000 | 0.5000 | 3.33e-16 | 1.43e-12 | 39 |
| 6 | 0.250000000 | 0.2500 | 6.11e-16 | 1.27e-12 | 39 |

And on a spectrum with a tight top, `1, 0.99, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015`:

| `k` | computed | true | value error | residual | steps |
|---|---|---|---|---|---|
| 1 | 1.000000000 | 1.0000 | 2.22e-16 | 9.93e-13 | 2285 |
| 2 | 0.990000000 | 0.9900 | 5.55e-16 | 1.33e-12 | 40 |
| 3 | 0.500000000 | 0.5000 | 5.55e-17 | 1.84e-12 | 41 |
| 4 | 0.250000000 | 0.2500 | 5.55e-17 | 1.39e-12 | 39 |
| 5 | 0.120000000 | 0.1200 | 6.94e-17 | 1.27e-12 | 39 |
| 6 | 0.060000000 | 0.0600 | 4.16e-17 | 1.30e-12 | 39 |

**The error does not accumulate.** It is flat at 1e-15 to 1e-17 across all six deflations, in both
spectra. This contradicts the common warning that deflation errors pile up, and the reason it does
is that the deflation is applied to a **symmetric** matrix using an eigenvector accurate to the
requested tolerance, so the remnant left behind is of size `tol` and the next power iteration
simply finds the next eigenvalue with the same accuracy.

What does change is the **cost**, and it changes at exactly one place. In the clustered spectrum
the first eigenvalue takes 2285 steps, because `|lambda_2/lambda_1| = 0.99`. Once `lambda_1` is
deflated away, the remaining ratios are all comfortable and every later step takes about 40
iterations. So clustering costs you time at the cluster and nothing afterwards, and it costs no
accuracy at all.

The split between the two sources is visible in the two columns. The value error is at the level
of `eps` times the eigenvalue, so it contributes nothing. The residual is at the level of the
requested tolerance, 1e-12, and that is the eigenvector contribution. The eigenvector term is
about a thousand times larger, so it is the one that would matter if either did, and it still does
not grow with `k`.

### 4.3 Which eigenvalue does RQI converge to?

A 3 by 3 symmetric matrix with spectrum `{1, 2, 5}`, from a grid of 3600 starting vectors covering
the unit sphere:

| eigenvalue | basin, fraction of starts |
|---|---|
| 1.0 | 12.9 percent |
| 2.0 | 66.9 percent |
| 5.0 | 20.2 percent |

**The middle eigenvalue takes two thirds of the sphere.** That is the counterintuitive result and
it is worth understanding. The first Rayleigh quotient of a random unit vector is a weighted
average of the three eigenvalues with weights `c_i^2` summing to 1, so it lands near the centre of
the spectrum far more often than near either end. RQI then converges to the eigenvalue nearest
that first shift, which is the middle one for most starts.

The basins are also not simple regions. Near the boundaries between them the structure is
fractal, which is characteristic of a cubically convergent iteration and is the same phenomenon
as Newton fractals for polynomial root finding. This is why RQI is never used on its own: it
converges very fast to **an** eigenvalue, and you do not get to choose which one. It is used as a
final polishing step once another method has isolated the target, which is exactly how lesson 37
uses it inside the QR algorithm as the Rayleigh shift.

### Level 5, advanced

### 5.1 Why inverse iteration is stable.

Wilkinson's argument, stated precisely.

Suppose we solve `(A - sigma I) y = x` in floating point, with `sigma` extremely close to an
eigenvalue `lambda`, so that `A - sigma I` is nearly singular and `kappa(A - sigma I)` is about
`1 / |lambda - sigma|`, which may be 1e15.

Standard backward error analysis of Gaussian elimination with partial pivoting says the computed
`y_hat` is the exact solution of

```
(A - sigma I + F) y_hat = x,     ||F|| <= c(n) * eps * ||A - sigma I||
```

for a modest `c(n)`. Rearranging, the residual is

```
r = x - (A - sigma I) y_hat = F y_hat,     so   ||r|| <= c(n) eps ||A - sigma I|| ||y_hat||
```

Now expand `y_hat` in the eigenvector basis, `y_hat = sum a_i v_i`. From
`(A - sigma I) y_hat = x - r`,

```
sum a_i (lambda_i - sigma) v_i = x - r
```

so `a_i = (component of x - r along v_i) / (lambda_i - sigma)`. The denominator for the target
eigenvalue is tiny, of size `|lambda - sigma|`, and every other denominator is of size the gap. So

```
|a_target| / |a_other|  is about  gap / |lambda - sigma|
```

which is enormous. **The computed `y_hat` is dominated by the target eigenvector**, and the more
ill conditioned the solve, the more completely it is dominated. The forward error of the solve is
huge, and it is huge in exactly the direction we want. After normalising, `y_hat / ||y_hat||` is
an excellent eigenvector.

Put another way: the quantity `||y_hat||` being enormous is not a symptom of failure, it is the
signal that `sigma` is close to an eigenvalue, and the direction of that enormous vector is the
answer.

**The one case where it fails.** When the target eigenvalue is part of a cluster, or exactly
repeated, the argument breaks at the last step. If `lambda_1` and `lambda_2` are both within
`|lambda - sigma|` of the shift, then `a_1` and `a_2` are both enormous and the computed vector is
an arbitrary combination of `v_1` and `v_2`, determined by rounding. Running inverse iteration
twice with different starts gives two vectors from the same subspace that are **not orthogonal**,
and the eigenvector matrix loses orthogonality.

What implementations do about it: LAPACK's `dstein` reorthogonalizes, by explicitly applying
modified Gram-Schmidt against the previously computed vectors of the same cluster. That costs
`O(n * cluster^2)` and, for a spectrum that is one big cluster, degrades to `O(n^3)` with a large
constant. The modern answer is the MRRR algorithm, which sidesteps the whole issue by computing a
different **representation** of the matrix for each cluster, chosen so that within that
representation the cluster members are relatively well separated. That is exercise 38.5.2.

### 5.2 Simultaneous iteration is the QR algorithm.

Let `Q_0 = I` and define simultaneous iteration by

```
Z_k = A Q_{k-1},     Q_k R_k = Z_k   (thin QR)
```

and the unshifted QR algorithm by

```
A_0 = A,     A_{k-1} = U_k S_k  (QR),     A_k = S_k U_k
```

**Claim:** `A_k = Q_k^T A Q_k`, and `Q_k = U_1 U_2 ... U_k`.

Proof by induction. At `k = 0` both are trivial. Suppose `Q_{k-1} = U_1 ... U_{k-1}` and
`A_{k-1} = Q_{k-1}^T A Q_{k-1}`. Then

```
A Q_{k-1} = Q_{k-1} A_{k-1} = Q_{k-1} U_k S_k
```

The left side is `Z_k`, and the right side is a product of an orthogonal matrix `Q_{k-1} U_k` and
an upper triangular `S_k`, so by uniqueness of the QR factorization with positive diagonal,
`Q_k = Q_{k-1} U_k = U_1 ... U_k` and `R_k = S_k`. Then

```
Q_k^T A Q_k = U_k^T Q_{k-1}^T A Q_{k-1} U_k = U_k^T A_{k-1} U_k = U_k^T U_k S_k U_k = S_k U_k = A_k
```

which completes the induction.

**What this buys before lesson 37 derives anything.** Simultaneous iteration is power iteration
applied to a whole basis at once, so the `j`-th column converges to the `j`-th eigenvector at rate
governed by the ratio of neighbouring eigenvalues. Specifically the subspace spanned by the first
`j` columns converges at rate `|lambda_{j+1} / lambda_j|`. Since `A_k = Q_k^T A Q_k` and the
columns of `Q_k` are converging to the eigenvectors in order, `A_k` is converging to upper
triangular form, and the entry `A_k[j+1, j]` just below the diagonal is the measure of how far the
first `j` columns are from being invariant. So

```
|A_k[j+1, j]|  decays like  |lambda_{j+1} / lambda_j|^k
```

That is the QR algorithm's convergence rate, obtained with no new analysis, and exercise 37.4.1
measures it directly and confirms it.

It also explains immediately why shifts are essential. The rate for the bottom entry is
`|lambda_n / lambda_{n-1}|`, which is close to 1 whenever the two smallest eigenvalues are close,
and the fix is to shift so that the target is isolated, which is exactly what the Wilkinson shift
does.

### 5.3 The two sided Rayleigh quotient.

For a non-symmetric `A`, carry both a right vector `x` and a left vector `y` and use

```
lambda = y^H A x / y^H x
```

By the argument of 36.2.2 this is second order accurate, because both cross terms now cancel: `y`
approximates the left eigenvector, so `y^H A = lambda y^H` to first order, killing the term that
survived in the one sided case.

```python
def two_sided_rqi(A, tol=1e-14, max_iter=60, rng=None):
    """Two solves per step, with the SAME matrix A - lambda I, so one factorization serves both.
    About 1.3x the work of the one sided version for a much better rate on a non-normal matrix."""
    A = np.atleast_2d(np.asarray(A, dtype=complex))
    n = A.shape[0]
    g = np.random.default_rng() if rng is None else rng
    x = g.standard_normal(n) + 1j * g.standard_normal(n)
    y = g.standard_normal(n) + 1j * g.standard_normal(n)
    x, y = x / np.linalg.norm(x), y / np.linalg.norm(y)
    for _ in range(int(max_iter)):
        d = np.conj(y) @ x
        if abs(d) < 1e-300:
            break
        lam = complex((np.conj(y) @ (A @ x)) / d)
        if float(np.linalg.norm(A @ x - lam * x)) <= tol:
            break
        M = A - lam * np.eye(n)
        try:
            xn, yn = np.linalg.solve(M, x), np.linalg.solve(M.conj().T, y)
        except np.linalg.LinAlgError:
            break
        x, y = xn / np.linalg.norm(xn), yn / np.linalg.norm(yn)
    return lam
```

Measured residual trails, 8 by 8:

| matrix | one sided steps | two sided steps |
|---|---|---|
| symmetric | 6 | 8 |
| mildly non-normal, `diag(1..8) + 0.5 triu` | 7 | 8 |
| strongly non-normal, `diag(1..8) + 12 triu` | 14, stagnates | 10, converges |

One sided trail on the strongly non-normal matrix: `1.3e1, 4.1e-1, 2.0e-1, 1.5e-1, 3.8e-2,
1.7e-2, 2.3e-2`. It stops making progress and starts going back up.

Two sided trail on the same matrix: `1.6e1, 5.1e0, 6.7e-1, 9.3e-1, 9.1e-1, 6.7e-1, 1.4e-1`, and it
reaches the tolerance at step 10.

**The honest reading.** On a symmetric matrix the two sided version is a waste: it costs an extra
solve per step and takes 8 steps where the one sided takes 6, because the second vector adds
nothing when `y = x`. On a mildly non-normal matrix it is a wash. On a strongly non-normal matrix
the one sided version stagnates and the two sided one converges, which is where the extra solve
earns its cost.

The convergence order is hard to measure honestly here. With cubic convergence the iteration
passes from 1e-2 to 1e-16 in two steps, so there are only two or three points in the asymptotic
regime and any fitted exponent is dominated by where you start the fit. On the symmetric case the
one sided trail `1.4e-2, 6.7e-7, 6.9e-16` gives a three point exponent of 2.09, while the digit
counts 1.85, 6.17, 15.16 give ratios 3.3 and 2.5. The order is somewhere between 2 and 3 and the
data cannot pin it down more precisely than that. What is unambiguous is the step count, and the
step counts are in the table.

The cost is two solves per step against one, but both use the same matrix `A - lambda I`, so a
single LU factorization serves both, and the second solve is a triangular solve pair costing
`O(n^2)` against the `O(n^3)` factorization. In practice the overhead is about 30 percent, not
100 percent.

---

## Lesson 37, The QR Algorithm

### Level 1, conceptual

### 1.1 Why reduce to Hessenberg form first?

Because the reduction is finite and it makes every subsequent step cheap. A full QR step on a
dense matrix costs `O(n^3)`. On a Hessenberg matrix it costs `O(n^2)`, because there are only
`n - 1` subdiagonal entries to eliminate and a Givens rotation kills each one. Since the algorithm
needs `O(n)` steps, the total goes from `O(n^4)` to `O(n^3)`.

The reduction itself costs `O(n^3)` once, by Householder reflectors applied from both sides. The
essential fact is that **Hessenberg form is preserved by a QR step**: if `H` is Hessenberg and
`H = QR`, then `RQ` is Hessenberg again. So you pay the reduction once and every step afterwards
is cheap.

For a symmetric matrix, Hessenberg form is tridiagonal, and a step costs `O(n)`.

### 1.2 What does the shift do?

It changes the convergence rate of the bottom subdiagonal entry from `|lambda_n / lambda_{n-1}|`,
which can be arbitrarily close to 1, to `|lambda_n - mu| / |lambda_{n-1} - mu|`, which is tiny
when `mu` is close to `lambda_n`. Since the shift is recomputed each step from the current bottom
corner, and the bottom corner is converging to `lambda_n`, the shift becomes an ever better
estimate and the convergence becomes superlinear.

The Wilkinson shift takes the eigenvalue of the trailing 2 by 2 block that is closer to the corner
entry. That choice, rather than the corner entry itself, is what gives global convergence on
symmetric matrices and cubic local convergence.

### 1.3 What is deflation and why does it matter?

When a subdiagonal entry becomes negligible, the matrix splits into two blocks that can be worked
on independently, and the eigenvalues of the whole are the eigenvalues of the parts. In practice
the bottom entry converges first, so one eigenvalue is read off the corner, the active block
shrinks by one, and the work per step drops.

It matters for two reasons. It is what makes the total cost `O(n^3)` rather than `O(n^3)` per
eigenvalue: each eigenvalue takes about two steps, and each step on the active block of size `m`
costs `O(m^2)`, so summing over `m` from `n` down to 1 gives `O(n^3)` overall. And it is what
allows the algorithm to terminate at all, since without deflation the iteration would keep
refining eigenvalues that are already exact.

### Level 2, mathematical

### 2.1 Prove that a QR step is a similarity transformation, and that it preserves Hessenberg form.

If `A = QR` with `Q` orthogonal, then `R = Q^T A`, so

```
A' = R Q = Q^T A Q
```

which is an orthogonal similarity. Eigenvalues are preserved by 35.2.4, and orthogonality means no
amplification of existing errors.

For the Hessenberg property, suppose `H` is upper Hessenberg, so `h_ij = 0` for `i > j + 1`. The
QR factorization of `H` by Givens rotations uses rotations acting on rows `(k, k+1)` for
`k = 1..n-1`, each clearing `h_{k+1,k}`. The product of these is `Q^T`, and `Q` is therefore a
product of Givens rotations on adjacent index pairs, which makes `Q` itself upper Hessenberg.

Now `A' = R Q` is the product of an upper triangular matrix and an upper Hessenberg matrix. For
such a product, `(RQ)_{ij} = sum_k R_{ik} Q_{kj}`, and `R_{ik} = 0` for `k < i` while
`Q_{kj} = 0` for `k > j + 1`. So the sum is empty unless `i <= j + 1`, which is exactly upper
Hessenberg.

### 2.2 Prove that the unshifted QR algorithm converges, and give the rate.

This follows from exercise 36.5.2, which shows the QR algorithm is simultaneous iteration in
disguise: `A_k = Q_k^T A Q_k` where `Q_k` comes from orthonormalising `A^k`.

Simultaneous iteration converges when the eigenvalues have distinct magnitudes,
`|lambda_1| > |lambda_2| > ... > |lambda_n|`, and the leading principal submatrices of the
eigenvector matrix are nonsingular. Under those hypotheses the span of the first `j` columns of
`Q_k` converges to the invariant subspace of the top `j` eigenvalues at rate
`|lambda_{j+1}/lambda_j|`, so

```
|A_k[j+1, j]|  =  O( |lambda_{j+1} / lambda_j|^k )
```

and `A_k` converges to upper triangular form, giving the Schur form in the limit.

The rate is what makes the unshifted algorithm useless in practice: if any two neighbouring
eigenvalues are close in magnitude, the corresponding subdiagonal entry decays arbitrarily slowly,
and if two have equal magnitude it does not decay at all.

### 2.3 Prove that the Wilkinson shift gives cubic convergence on a symmetric matrix.

For a symmetric tridiagonal matrix, a shifted QR step with shift `mu` is equivalent to one step of
inverse iteration with shift `mu` applied to the last column, followed by re-orthonormalisation.
The Wilkinson shift takes `mu` to be the eigenvalue of the trailing 2 by 2 block

```
[ a_{n-1}   b_{n-1} ]
[ b_{n-1}   a_n     ]
```

closer to `a_n`.

Let `e = |b_{n-1}|` be the size of the bottom off diagonal entry, the quantity we want to see go
to zero. A perturbation analysis of the 2 by 2 block shows that the Wilkinson shift satisfies

```
|mu - lambda_n|  =  O(e^2)
```

because the block's eigenvalue differs from the true `lambda_n` by an amount governed by how much
the rest of the matrix couples in, which is second order in the coupling `e`.

One QR step with shift `mu` scales the bottom off diagonal entry by roughly

```
e_new  is about  e * |mu - lambda_n| / gap
```

by the inverse iteration rate of 36.2.5. Substituting `|mu - lambda_n| = O(e^2)` gives

```
e_new = O(e^3)
```

which is cubic. The gap is bounded below by the separation of `lambda_n` from the rest of the
spectrum and contributes only a constant.

### 2.4 Prove that a real matrix has a real Schur form.

The complex Schur form of 35.2.5 uses complex unitary transformations, which a real
implementation cannot use. The real version replaces the requirement "upper triangular" with
"quasi upper triangular": block upper triangular with 1 by 1 and 2 by 2 diagonal blocks, where each
2 by 2 block has a complex conjugate pair of eigenvalues.

Proof by induction on `n`, mirroring 35.2.5. If `A` has a real eigenvalue `lambda`, take a real
unit eigenvector `x`, extend to a real orthogonal `U_1` with `x` as first column, and

```
U_1^T A U_1 = [ lambda   w^T ]
              [   0      A_1 ]
```

with `A_1` real of size `n - 1`, and apply induction.

If `A` has no real eigenvalue, take a complex pair `alpha +- i beta` with `beta != 0` and an
eigenvector `x + i y`. Then `span{x, y}` is a real two dimensional invariant subspace, since
`A(x + iy) = (alpha + i beta)(x + iy)` gives `A x = alpha x - beta y` and `A y = beta x + alpha y`.
Orthonormalise `{x, y}` to get the first two columns of a real orthogonal `U_1`, and

```
U_1^T A U_1 = [ B    W ]
              [ 0   A_1 ]
```

with `B` a 2 by 2 real block having eigenvalues `alpha +- i beta`, and `A_1` real of size `n - 2`.
Apply induction.

This is why the practical algorithm is the **implicit double shift**: it takes a complex conjugate
pair of shifts using only real arithmetic, and converges to the real Schur form directly.

### 2.5 Prove that the LR algorithm can be unstable and the QR algorithm cannot.

The LR algorithm factors `A = LU` and forms `A' = UL = L^{-1} A L`. This is a similarity, so the
eigenvalues are preserved, and it is cheaper than QR by a factor of about 2.

The instability is that `L` from an unpivoted LU factorization can be arbitrarily ill conditioned,
and `kappa(L)` multiplies into the transformed matrix at every step. If `E` is the rounding error
present in `A`, then after the transformation it becomes `L^{-1} E L`, whose norm can be as large
as `kappa(L) ||E||`. Over many steps these factors compound, and there is no bound on the product.

The QR algorithm uses `A' = Q^T A Q` with `Q` orthogonal, so `||Q^T E Q||_2 = ||E||_2` exactly. The
error present is transported without amplification, at every step, forever. This is the entire
argument for orthogonal transformations, and it is what exercise 37.4.3 measures.

Adding pivoting to LR bounds `kappa(L)` by `2^{n-1}` in the worst case, which is a bound but not a
useful one, and pivoting destroys the Hessenberg structure that made the steps cheap.

### Level 3, computational

### 3.1 A Hessenberg QR step by Givens rotations.

```python
def hessenberg_qr_step_givens(H, mu=0.0):
    """A Hessenberg matrix has only n-1 subdiagonal entries, so n-1 Givens rotations bring it to
    triangular form: O(n^2), not O(n^3). Applying them back on the right restores Hessenberg
    form. numpy.linalg.qr does not know the structure and pays the full O(n^3)."""
    H = np.array(H, dtype=float, copy=True)
    n = H.shape[0]
    H[np.diag_indices(n)] -= mu
    cs = np.empty(n - 1); sn = np.empty(n - 1)
    for k in range(n - 1):
        c, s = qrmod.givens_rotation(H[k, k], H[k + 1, k])
        cs[k], sn[k] = c, s
        rows = H[k:k + 2, k:].copy()
        H[k, k:] = c * rows[0] + s * rows[1]
        H[k + 1, k:] = -s * rows[0] + c * rows[1]
    for k in range(n - 1):
        c, s = cs[k], sn[k]
        cols = H[:, k:k + 2].copy()
        H[:, k] = c * cols[:, 0] + s * cols[:, 1]
        H[:, k + 1] = -s * cols[:, 0] + c * cols[:, 1]
    H[np.diag_indices(n)] += mu
    return H
```

Measured against `numpy.linalg.qr` followed by `R @ Q`. The comparison has to be made on
**eigenvalues**, not on entries: the QR factorization is unique only up to the signs of `R`'s
diagonal, so two correct implementations produce results differing by a diagonal sign similarity.

| `n` | Givens | numpy qr route | speedup | eigenvalues agree to | Hessenberg preserved |
|---|---|---|---|---|---|
| 20 | 0.00030s | 0.00007s | 0.22x | 1.34e-14 | True |
| 50 | 0.00077s | 0.00010s | 0.13x | 4.36e-14 | True |
| 100 | 0.00170s | 0.00034s | 0.20x | 4.94e-14 | True |
| 200 | 0.00377s | 0.00297s | 0.79x | 1.74e-13 | True |
| 400 | 0.00903s | 0.01475s | 1.63x | 4.33e-13 | True |
| 800 | 0.02362s | 0.05759s | 2.44x | 6.71e-13 | True |

**The crossover is near `n = 250`.** Below it the Python loop overhead dominates and the LAPACK
call wins despite doing far more arithmetic. Above it the asymptotics take over: 1.63x at
`n = 400` and 2.44x at `n = 800`.

The timings confirm the complexity. The Givens times go 0.30, 0.77, 1.70, 3.77, 9.03, 23.62 in
units of 1e-3, roughly doubling as `n` doubles, which is `O(n)` numpy calls of `O(n)` work each and
so `O(n^2)` flops. The numpy times go 0.07, 0.10, 0.34, 2.97, 14.75, 57.59, growing by about 5x per
doubling above `n = 100`, consistent with `O(n^3)`.

The lesson is not that this Python loop beats LAPACK. It is that the **structure** is worth a
factor of `n`, and any implementation that throws it away pays that factor. A compiled Givens
loop, as in LAPACK's `dhseqr`, wins at every size.

### 3.2 The implicit double shift (Francis).

A real matrix with a complex conjugate pair of shifts `mu` and `conj(mu)` can take both in one
real step. The key identity is

```
(H - mu I)(H - conj(mu) I) = H^2 - s H + t I,     s = 2 Re(mu),  t = |mu|^2
```

with `s` and `t` both real, so the product is a real matrix and its first column determines a real
Householder reflector. Applying that reflector creates a 3 by 3 bulge below the subdiagonal, and
chasing the bulge down restores Hessenberg form. No complex number appears anywhere, and the
matrix `H^2` is never formed: only its first column is needed, and that costs `O(1)`.

```python
def _house(x):
    """v and beta with (I - beta v v^T) x = ||x|| e_1."""
    x = np.asarray(x, dtype=float)
    nx = float(np.linalg.norm(x))
    if nx == 0.0:
        return x, 0.0
    v = x.copy()
    v[0] += (1.0 if x[0] >= 0 else -1.0) * nx
    vv = float(v @ v)
    return v, (0.0 if vv == 0.0 else 2.0 / vv)


def francis_double_step(H):
    H = np.array(H, dtype=float, copy=True)
    n = H.shape[0]
    if n < 3:
        return H
    s = H[n - 2, n - 2] + H[n - 1, n - 1]
    t = H[n - 2, n - 2] * H[n - 1, n - 1] - H[n - 2, n - 1] * H[n - 1, n - 2]
    x = H[0, 0] * H[0, 0] + H[0, 1] * H[1, 0] - s * H[0, 0] + t
    y = H[1, 0] * (H[0, 0] + H[1, 1] - s)
    z = H[1, 0] * H[2, 1]
    for k in range(n - 2):
        v, beta = _house([x, y, z])
        if beta != 0.0:
            q = max(k - 1, 0)
            H[k:k + 3, q:] -= beta * np.outer(v, v @ H[k:k + 3, q:])
            r = min(k + 4, n - 1)                 # rows 0 through r, inclusive
            H[:r + 1, k:k + 3] -= beta * np.outer(H[:r + 1, k:k + 3] @ v, v)
        x, y = H[k + 1, k], H[k + 2, k]
        if k < n - 3:
            z = H[k + 3, k]
    v, beta = _house([x, y])
    if beta != 0.0:
        H[n - 2:n, n - 3:] -= beta * np.outer(v, v @ H[n - 2:n, n - 3:])
        H[:, n - 2:n] -= beta * np.outer(H[:, n - 2:n] @ v, v)
    return H
```

Measured against the explicit single shift, 12 by 12:

| matrix | complex eigenvalues | single shift steps | Francis steps | error |
|---|---|---|---|---|
| random real, mixed spectrum | 10 | 164 | 22 | 7.4e-14 |
| rotation blocks, all complex | 12 | 0 | 0 | 0 |
| companion of `x^n - 1` | 10 | 40000, failed | 10000, failed | 1.0 |

And the step count against `n` on random real matrices:

| `n` | single shift | Francis | Francis per eigenvalue | error |
|---|---|---|---|---|
| 8 | 94 | 18 | 2.25 | 4.9e-14 |
| 16 | 321 | 62 | 3.88 | 8.6e-14 |
| 32 | 858 | 62 | 1.94 | 1.7e-14 |
| 64 | 1845 | 138 | 2.16 | 2.6e-13 |

Francis takes **about two steps per eigenvalue** at every size, which is the folklore, and it is
5x to 13x fewer steps than the explicit single shift. The single shift wastes steps because on a
real matrix with complex eigenvalues a real shift can never be close to the target, so its
convergence rate stays poor until the block deflates to 2 by 2.

The companion matrix of `x^n - 1` defeats both, which is exercise 37.5.1's subject.

### 3.3 Eigenvector recovery from the Schur form.

`T` is upper triangular with the eigenvalues on its diagonal. For the `k`-th eigenvalue the
eigenvector of `T` is supported on its first `k` entries and solves a triangular system by back
substitution. Mapping back with `Q` gives the eigenvectors of `A`.

```python
def eigenvectors_from_schur(T, Q, tol=1e-30):
    """O(k^2) per vector, so O(n^3) for all of them, and one factorization is never needed
    because T is already triangular. Inverse iteration needs a fresh factorization per vector."""
    T = np.atleast_2d(np.asarray(T, dtype=float))
    n = T.shape[0]
    V = np.zeros((n, n))
    for k in range(n):
        w = np.zeros(n)
        w[k] = 1.0
        if k > 0:
            M = T[:k, :k] - T[k, k] * np.eye(k)
            d = np.diag(M).copy()
            d[np.abs(d) < tol] = tol       # a repeated eigenvalue makes a diagonal entry zero
            np.fill_diagonal(M, d)
            w[:k] = np.linalg.solve(M, -T[:k, k])
        v = Q @ w
        V[:, k] = v / np.linalg.norm(v)
    return V
```

Measured on symmetric matrices:

| `n` | Schur residual | inverse iteration residual | Schur time | inverse iteration time |
|---|---|---|---|---|
| 10 | 2.87e-14 | 2.73e-14 | 0.0010s | 0.0019s |
| 30 | 6.40e-14 | 4.54e-14 | 0.0010s | 0.0051s |
| 60 | 1.23e-13 | 1.17e-13 | 0.0030s | 0.0134s |
| 120 | 1.61e-13 | 1.45e-13 | 0.0083s | 0.0349s |

The accuracy is the same to within 10 percent, and the Schur route is 1.9x to 4.2x faster. The
reason is that inverse iteration pays a fresh `O(n^3)` factorization per eigenvector, so `n` of
them cost `O(n^4)`, while the triangular solves cost `O(n^3)` for all of them together. The
measured ratios understate this because `n` is small and LAPACK's factorization is very fast.

The one thing the Schur route needs care with is a repeated or nearly repeated eigenvalue, where
the triangular system becomes singular. The perturbation of the diagonal above is the simple fix
and it is what LAPACK's `dtrevc` does with more care, scaling to avoid overflow.

### Level 4, experimental

### 4.1 The decay rate of each subdiagonal entry.

Unshifted QR on a symmetric 6 by 6 with spectrum `8, 4, 2, 1, 0.5, 0.25`, tracking every
subdiagonal entry over 120 steps:

| entry | fitted rate | `\|lambda_{i+1}/lambda_i\|` | ratio | points used |
|---|---|---|---|---|
| (2,1) | 0.5011 | 0.5000 | 1.0023 | 40 |
| (3,2) | 0.4921 | 0.5000 | 0.9841 | 40 |
| (4,3) | 0.5179 | 0.5000 | 1.0358 | 40 |
| (5,4) | 0.4884 | 0.5000 | 0.9769 | 38 |
| (6,5) | 0.5023 | 0.5000 | 1.0047 | 40 |

Every rate is within 3.6 percent of the prediction, confirming 36.5.2 and 37.2.2 directly. The
scatter is what a least squares fit through 40 points of a noisy geometric decay gives. It is not
a systematic deviation, since the ratios sit on both sides of 1.

**Where the fit breaks down.** Spectrum `4, -4, 1, 0.5`, so `|lambda_2/lambda_1| = 1` exactly:

```
entry (2,1) started at 1.8555, and after 400 unshifted steps it is 1.7185
```

It does not decay at all. Two eigenvalues of equal magnitude and opposite sign give a ratio of
exactly 1, and the corresponding 2 by 2 block rotates forever without converging. The unshifted
algorithm has no mechanism to break the tie, and this is not a measure zero event: it happens for
every real matrix with a complex conjugate pair, since `|lambda|` and `|conj(lambda)|` are always
equal.

This is precisely why the real Schur form allows 2 by 2 blocks, and why the practical algorithm
uses the double shift of 37.3.2 rather than trying to force convergence to triangular form.

### 4.2 The total step count against `n` for each shift strategy.

| `n` | no shift | Rayleigh shift | Wilkinson shift |
|---|---|---|---|
| 4 | 98 | 11 | 7 |
| 8 | 202 | 23 | 18 |
| 16 | 403 | 53 | 35 |
| 32 | 895 | 114 | 68 |
| 64 | 1712 | 209 | 133 |
| 96 | 2559 | 338 | 198 |

Fitted exponents: no shift `n^1.031`, Rayleigh `n^1.075`, Wilkinson `n^1.026`.

Steps per eigenvalue at `n = 96`: no shift 26.66, Rayleigh 3.52, Wilkinson **2.06**.

**All three are linear in `n`.** That is the first result and it is easy to misread. Shifting does
not improve the exponent, it improves the **constant**, by a factor of 13 for Wilkinson against no
shift, and by a factor of 1.7 for Wilkinson against Rayleigh. Since each step on the active block
costs `O(m^2)` and the block shrinks, linear step count means `O(n^3)` total work, which is the
headline complexity of the QR algorithm.

The "two steps per eigenvalue" folklore is confirmed for the Wilkinson shift, at 2.06 measured, and
it is a good rule at every size from 4 to 96. The Rayleigh shift needs 3.52, which is worse but
still a constant, and the unshifted algorithm needs 26.66 and rising, since its per eigenvalue
count is not really constant but depends on the eigenvalue ratios.

### 4.3 Conditioning under LR against QR.

Tracking `kappa` of the accumulated transformation matrix at each step:

| matrix | LR, `kappa` start | LR, `kappa` end | LR max | LR steps | converged | QR `kappa` |
|---|---|---|---|---|---|---|
| `n = 4` | 2.716 | 3.115 | 3.263 | 147 | yes | 2.716, constant |
| `n = 6` | 2.838 | 7.472 | 7.472 | 400 | **no** | 2.838, constant |

**QR's conditioning is exactly constant**, to the last digit, at every step of both runs. That is
37.2.5 made visible: an orthogonal similarity has `kappa = 1` and composing orthogonal matrices
gives an orthogonal matrix, so the accumulated transformation never degrades.

LR's conditioning grows, and at `n = 6` it grew by a factor of 2.6 over 400 steps and the algorithm
**failed to converge at all**. The growth is not dramatic on these small well behaved matrices,
which is worth saying honestly: LR is not catastrophically unstable on a random matrix, it is
unreliably stable, and the failure mode is that `kappa` grows without bound on some matrices and
not others, with no way to know in advance which.

That unreliability is the whole argument. QR costs about twice as much per step and offers a
guarantee. LR is cheaper and offers a hope. The field chose the guarantee, and the modern dqds
algorithm of exercise 42.3.2 is a descendant of LR that recovered the stability by restricting to
positive quantities, which is the one setting where the `L` factor is automatically well behaved.

### Level 5, advanced

### 5.1 The shifted QR algorithm can still fail.

The classic failure is the cyclic permutation matrix, equivalently the companion matrix of
`x^n - 1`:

```
[ 0 0 ... 0 1 ]
[ 1 0 ... 0 0 ]
[ 0 1 ... 0 0 ]
[ ...         ]
[ 0 0 ... 1 0 ]
```

Its eigenvalues are the `n`-th roots of unity, all of modulus exactly 1.

Measured at `n = 12`: the explicit single shift ran 40000 steps without converging, and the Francis
double shift ran 10000 steps without converging. Both fail completely.

**The mechanism.** The Wilkinson shift is computed from the trailing 2 by 2 block. For this matrix
that block is

```
[ 0  0 ]
[ 1  0 ]
```

whose eigenvalues are both 0. So the shift is 0, the shifted QR step is an unshifted step, and the
unshifted algorithm cannot converge because every eigenvalue has the same modulus, by 37.2.2. The
matrix is invariant under the unshifted step up to a permutation, so the iteration cycles exactly
and makes no progress ever. It is not slow convergence, it is no convergence.

**What real implementations do.** After a fixed number of steps without deflation, typically 10 in
LAPACK's `dhseqr`, they apply an **exceptional shift**: a shift chosen arbitrarily, usually from
the size of the subdiagonal entries rather than from the trailing block. LAPACK uses
`mu = 3/4 * |h_{m,m-1}| + h_{m,m}` for the single shift case, and for the double shift it
constructs an exceptional pair from `|h_{m,m-1}| + |h_{m-1,m-2}|`. The point is that any shift
which breaks the symmetry of the situation will do, because the failure is caused by an exact
symmetry and an arbitrary perturbation destroys it. After one exceptional shift the matrix is no
longer a cyclic permutation and the normal shifts work.

LAPACK also caps the total steps and returns an error code if convergence fails, so a caller is
never left in an infinite loop.

### 5.2 Why the implicit form is used.

The explicit shifted step computes `A - mu I`, factors it, and forms `RQ + mu I`. The numerical
objection is to the first and last operations when `mu` is close to an eigenvalue.

If `mu` is accurate to `eps` relative to `||A||`, then `A - mu I` has an eigenvalue of size about
`eps ||A||`, so `A - mu I` is numerically singular. Forming it explicitly loses the information
that distinguishes that tiny eigenvalue from zero: the entries of `A - mu I` near the corner are
differences of nearly equal numbers, and the relative accuracy of that difference is destroyed.
Adding `mu I` back at the end does not restore what was lost, because the information was
discarded in the subtraction, not in the factorization.

Concretely, if `a_nn = 1 + 1e-16` and `mu = 1`, the difference is computed as `1e-16` with **zero**
correct digits, because `1 + 1e-16` was already rounded to `1` on the way in.

The implicit form never forms `A - mu I`. It computes only the first column of `(A - mu I) e_1`,
or of `(A - mu I)(A - conj(mu) I) e_1` for the double shift, determines a reflector from that
short vector, applies it, and chases the resulting bulge. The bulge chase uses only the entries of
`A` itself. The shift enters through one small vector and never through a subtraction of the full
matrix.

This is exactly the argument of 36.1.3 and 36.5.1 in a different costume. There, the deliberately
singular solve was safe because the error pointed along the wanted direction. Here, the shift is
kept out of the matrix so that the near singularity is never materialised at all. Both are ways of
handling a near singular object without letting cancellation destroy it, and both are why the
practical algorithms look more complicated than the textbook description.

### 5.3 Aggressive early deflation.

The classical deflation test looks only at the single subdiagonal entry `h_{m,m-1}` and deflates
when it is negligible. Braman, Byers and Mathias observed that far more eigenvalues are actually
converged than that test finds, and that they can be detected by looking at a **window**.

The idea: take the trailing `w` by `w` block, with `w` typically a few times the number of shifts.
Compute its Schur form. The coupling of that window to the rest of the matrix is a single column,
the **spike**, formed by `h_{m-w, m-w-1}` times the last row of the window's Schur vectors. Any
eigenvalue of the window whose corresponding spike entry is negligible is converged and can be
deflated, even though the classical test on `h_{m,m-1}` would not have found it. Reorder the
window's Schur form to move the converged ones to the bottom, deflate them all at once, and
continue.

```python
def aggressive_early_deflation(H, window, tol=1e-13):
    """Deflate everything in the trailing window whose spike entry is negligible, not just the
    last one. Returns the updated matrix and how many eigenvalues were removed."""
    H = np.array(H, dtype=float, copy=True)
    n = H.shape[0]
    w = min(int(window), n - 1)
    if w < 1:
        return H, 0
    lo = n - w
    T, Z = sla.schur(H[lo:, lo:])
    spike = H[lo, lo - 1] * Z[0, :] if lo > 0 else np.zeros(w)
    scale = max(float(np.linalg.norm(H, 1)), 1e-300)
    keep = np.abs(spike) > tol * scale
    n_deflated = int(np.sum(~keep))
    if n_deflated == 0:
        return H, 0
    order = np.argsort(keep)[::-1]            # the still coupled ones first
    T, Z = T[np.ix_(order, order)], Z[:, order]
    H[lo:, lo:] = T
    if lo > 0:
        H[lo:, :lo] = Z.T @ H[lo:, :lo]
        H[:lo, lo:] = H[:lo, lo:] @ Z
    return H, n_deflated
```

What it finds, measured on random Hessenberg matrices after running `2n/3` shifted QR steps,
with a window of `w = min(max(n/4, 4), 32)`:

| `n` | window | QR steps run | classical test finds | window test finds | extra |
|---|---|---|---|---|---|
| 32 | 8 | 21 | 2 | 2 | 0 |
| 64 | 16 | 42 | 1 | 7 | 6 |
| 128 | 32 | 85 | 0 | 22 | 22 |
| 256 | 32 | 170 | 1 | 19 | 18 |

At `n = 128` the classical test finds **nothing** while the window test finds 22 converged
eigenvalues. That is the point: 22 eigenvalues were already converged and the single entry test
could not see any of them. At `n = 32` the window is too small for there to be anything extra to
find, and both agree.

Each extra eigenvalue found is a QR step not taken, and each avoided step on a block of size `m`
saves `O(m^2)` work. The reason it works is that the classical test is a **sufficient** condition
for deflation, not a necessary one. An eigenvalue can be fully converged while `h_{m,m-1}` is not
small, if the convergence happened in a direction the single entry does not see. The window test
measures the actual coupling in the Schur basis, which is the right quantity.

---
## Lesson 38, The Symmetric Eigenvalue Problem

### Level 1, conceptual

### 1.1 Symmetry buys four things. Which one makes the tridiagonal reduction possible?

One, the eigenvalues are **real**, so there is no complex arithmetic and no 2 by 2 blocks in the
Schur form.

Two, the eigenvectors are **orthogonal**, so the eigendecomposition is `A = Q D Q^T` with `Q`
orthogonal, and `kappa(Q) = 1`.

Three, the eigenvalues are **perfectly conditioned**, every condition number exactly 1 by 35.2.3,
so a backward stable method gives eigenvalues accurate to `eps ||A||` with no amplification.

Four, the Schur form **is** the eigendecomposition, since a symmetric triangular matrix is
diagonal. There is no gap between the numerically sound target and the mathematically desired one.

The one that makes the tridiagonal reduction possible is the **second**. Hessenberg reduction by
orthogonal similarity gives `Q^T A Q`, which is symmetric whenever `A` is, because
`(Q^T A Q)^T = Q^T A^T Q = Q^T A Q`. A symmetric Hessenberg matrix is tridiagonal. So symmetry is
preserved by exactly the transformations the algorithm uses, and tridiagonal form comes for free.

The consequence is large: storage drops from `n^2` to `2n`, a QR step drops from `O(n^2)` to
`O(n)`, and the whole eigenvalue computation drops from `O(n^3)` to `O(n^2)` once the reduction
is done, with the reduction itself the only `O(n^3)` part.

### 1.2 A norm computed as `sqrt(||A||^2 - ||diag||^2)` passed every eigenvalue test and failed the eigenvector test. Why did the eigenvalues survive?

The quantity is the off diagonal norm, used as Jacobi's stopping criterion. Computing it as a
difference of two nearly equal numbers gives it a roundoff floor of about `sqrt(eps) ||A||`, by
the same argument as in 35.5.3. So the criterion reports zero while the true off diagonal norm is
still about `1e-8 ||A||`, and Jacobi stops several sweeps early.

The eigenvalues survive because they are **second order accurate** in the off diagonal norm. For a
symmetric matrix, the diagonal of a nearly diagonal matrix differs from the eigenvalues by
`O(off^2 / gap)`. With `off` about `1e-8`, that is about `1e-16`, which is at machine precision.
So the eigenvalues are perfect.

The eigenvectors are only **first order** accurate. They differ from the true eigenvectors by
`O(off / gap)`, which is about `1e-8`, and the measured error in this repository was 3e-9, matching.

That asymmetry is the whole lesson. A stopping test that is adequate for the values is inadequate
for the vectors by a factor of `1 / off`, and the only test that catches it is the residual
`||A V - V D||`, which involves the vectors. This bug was found in `nalib.symeig.off_norm` and
fixed by computing the off diagonal norm directly instead of by subtraction.

### 1.3 Bisection can find eigenvalue 37 of 1000 without the other 999. Why can neither Jacobi nor the QR algorithm do that?

Because bisection is built on a **counting** function, not on a transformation. The Sturm count
tells you how many eigenvalues lie below `x`, computed by one `O(n)` pass over the tridiagonal
matrix. To find eigenvalue 37 you bisect on `x` until the count crosses 37, which costs about 50
passes for full precision, so `O(n)` work total and no dependence on the other eigenvalues at all.

Jacobi and the QR algorithm both work by **transforming the whole matrix** toward diagonal form.
Every rotation touches two full rows and columns, and the process is not finished until every off
diagonal entry is negligible. There is no way to ask them for one eigenvalue: the intermediate
matrix does not contain any single eigenvalue until it contains nearly all of them. Deflation
gives them up one at a time from the bottom, but only in that order and only after the work of
converging them.

The practical consequence is that when you want a few eigenvalues from a large symmetric matrix,
bisection with inverse iteration costs `O(n k)` and the transformation methods cost `O(n^3)`. For
`k` small and `n` large that is the entire difference between feasible and not.

### Level 2, mathematical

### 2.1 Show that a Jacobi rotation reduces `off(A)^2` by exactly `2 a_pq^2`.

Define `off(A)^2 = sum over i != j of a_ij^2`, and note `||A||_F^2 = sum a_ii^2 + off(A)^2`.

A Jacobi rotation `J` in the `(p, q)` plane gives `A' = J^T A J`, which is an orthogonal
similarity, so `||A'||_F = ||A||_F`. The rotation is chosen to make `a'_pq = 0`.

The rotation changes only rows and columns `p` and `q`. For any `i` outside `{p, q}`, the pair
`(a_ip, a_iq)` is rotated, so `a'^2_ip + a'^2_iq = a^2_ip + a^2_iq`. Those contributions to
`off^2` are unchanged. The same holds for the columns by symmetry.

So the only change is in the 2 by 2 block. There, `a'^2_pp + a'^2_qq = a^2_pp + a^2_qq + 2 a^2_pq`,
because the block's Frobenius norm is preserved and `a'_pq = a'_qp = 0` removes `2 a^2_pq` from the
off diagonal part. Therefore

```
off(A')^2 = ||A||_F^2 - sum a'^2_ii = off(A)^2 - 2 a_pq^2
```

For convergence, choose `(p, q)` to be the largest off diagonal entry. Then
`a_pq^2 >= off(A)^2 / (n(n-1))`, since there are `n(n-1)` off diagonal entries, so

```
off(A')^2 <= off(A)^2 ( 1 - 2/(n(n-1)) )
```

which is a strict geometric decrease, and `off` goes to zero. Convergence is therefore guaranteed
for the greedy ordering, for any symmetric matrix, with no hypotheses at all.

### 2.2 Prove the quadratic convergence of cyclic Jacobi.

Suppose after some sweep `off(A) = e` and `e` is smaller than `d/2`, where `d` is the smallest gap
between distinct eigenvalues. Then each diagonal entry `a_ii` is within `O(e^2/d)` of a distinct
eigenvalue, by the second order accuracy noted in 38.1.2, so the diagonal entries are separated by
at least `d - O(e^2/d)`, which is about `d`.

Now consider one full sweep of `n(n-1)/2` rotations. When rotation `(p, q)` is applied, it
annihilates `a_pq`. Later rotations in the same sweep can make `a_pq` nonzero again, but only by
the amount they bring in from other entries, and the transfer coefficient is governed by the
rotation angle. The rotation angle for the pair `(r, s)` satisfies

```
tan(2 theta) = 2 a_rs / (a_rr - a_ss)
```

so `|theta| <= |a_rs| / |a_rr - a_ss|`, which is `O(e/d)` since `|a_rs| <= e` and the diagonal
separation is about `d`. A rotation through an angle `theta` transfers at most `|theta|` times the
magnitude of the entries it touches into `a_pq`.

Therefore after the sweep, each `a_pq` is at most `O(e/d)` times `O(e)`, that is `O(e^2/d)`, and

```
off(A_new)  =  O( off(A)^2 / d )
```

which is quadratic convergence. The measured consequence is that once Jacobi gets close, it
finishes in two or three more sweeps regardless of `n`, which is exactly what exercise 38.4.3
observes.

### 2.3 Prove the Sturm count equals the number of eigenvalues below `x`.

Sylvester's law of inertia: if `M` is symmetric and `S` is nonsingular, then `S^T M S` has the same
number of positive, negative and zero eigenvalues as `M`. The proof is a continuity argument on
the path from `S` to the identity through nonsingular matrices, along which no eigenvalue can
cross zero.

Now factor `T - x I = L D L^T` with `L` unit lower triangular and `D` diagonal. This is the
`LDL^T` factorization, and it exists as long as no leading principal minor vanishes. By Sylvester
with `S = L^{-T}`, the matrix `T - x I` has the same inertia as `D`. So

```
number of negative entries of D  =  number of negative eigenvalues of T - x I
                                 =  number of eigenvalues of T strictly below x
```

For a tridiagonal `T` with diagonal `a` and off diagonal `b`, the `LDL^T` recursion is

```
d_1 = a_1 - x
d_i = a_i - x - b_{i-1}^2 / d_{i-1}
```

so one `O(n)` pass gives all of `D`, and counting the negative `d_i` gives the answer. That is the
Sturm count.

The one practical care needed is that `d_{i-1}` can be zero, which would divide by zero. The
standard fix is to replace an exact zero by a tiny value of the right sign, which changes the
matrix by an amount below the rounding level and does not change the count.

### 2.4 Derive the secular equation, and prove the interlacing.

Divide and conquer splits the tridiagonal `T` into two halves plus a rank one correction:

```
T = [ T_1  0  ]  +  rho * v v^T
    [ 0   T_2 ]
```

where `v` has ones only in the two positions straddling the split. Solve the halves recursively,
giving `T_1 = Q_1 D_1 Q_1^T` and `T_2 = Q_2 D_2 Q_2^T`. Then with `Q = diag(Q_1, Q_2)` and
`z = Q^T v`,

```
Q^T T Q = D + rho z z^T,     D = diag(D_1, D_2)
```

so the problem reduces to the eigenvalues of a diagonal matrix plus a rank one update.

For the secular equation, `(D + rho z z^T) u = lambda u` gives `(D - lambda I) u = -rho (z^T u) z`,
so `u = -rho (z^T u) (D - lambda I)^{-1} z` provided `lambda` is not a `d_i`. Multiplying by `z^T`
and dividing by `z^T u`, which is nonzero when `z` has no zero entries,

```
1 + rho sum_i z_i^2 / (d_i - lambda)  =  0
```

which is the secular equation `f(lambda) = 0`.

For the interlacing, take `rho > 0` and order `d_1 < d_2 < ... < d_n`. Between consecutive `d_i`
the function `f` is continuous, and

```
f'(lambda) = rho sum_i z_i^2 / (d_i - lambda)^2  >  0
```

so `f` is strictly increasing on each interval. As `lambda` approaches `d_i` from above, the term
`z_i^2/(d_i - lambda)` goes to minus infinity, so `f` goes to minus infinity; as `lambda`
approaches `d_{i+1}` from below, that term goes to plus infinity. So `f` crosses zero exactly once
in each open interval `(d_i, d_{i+1})`, and once more above `d_n`. That gives `n` roots with

```
d_1 < lambda_1 < d_2 < lambda_2 < ... < d_n < lambda_n
```

which is the interlacing. Each root is bracketed by two known numbers, so a safeguarded root
finder cannot go wrong, and that is what makes the method reliable.

### 2.5 State the Demmel-Veselic theorem, and explain what its hypothesis rules out.

Write `A = D B D` with `D` diagonal positive and `B` having unit diagonal, so `B` is `A` scaled to
have ones on its diagonal. `kappa(B)` is called the **scaled condition number**.

**Theorem (Demmel and Veselic).** One sided Jacobi computes every eigenvalue of a symmetric
positive definite `A` with relative error bounded by `O(n) * eps * kappa(B)`, independently of
`kappa(A)`.

The point is that `kappa(A)` can be `kappa(B)` times `kappa(D)^2`, which for a badly scaled matrix
is astronomically larger. The QR algorithm's error bound is `eps kappa(A)` for the small
eigenvalues in a relative sense, so on a badly scaled matrix Jacobi has a guarantee and QR does
not.

**Where the hypothesis bites.** The theorem says nothing when `B` itself is badly conditioned.
That is not a technicality: `kappa(B)` badly conditioned means the matrix is ill conditioned in a
way that scaling cannot remove, so there is genuinely no information about the small eigenvalues
in the entries of `A`, at any precision. No algorithm can recover what the data does not contain.
The theorem is sharp in that sense: it says Jacobi extracts everything the scaling leaves
available, and exercise 38.4.2 measures that the error really does follow `kappa(B)` and not
`kappa(A)`.

### Level 3, computational

### 3.1 The tridiagonal QR algorithm with Givens rotations.

```python
def tridiagonal_qr(diag, off, tol=1e-14, max_iter=10_000):
    """One step costs O(n) instead of O(n^2), because a tridiagonal matrix has n-1 entries to
    clear and one Givens rotation clears each. Only the two diagonals are ever stored."""
    d = np.asarray(diag, dtype=float).copy()
    e = np.asarray(off, dtype=float).copy()
    n = d.size
    high = n - 1
    steps = 0
    while high > 0 and steps < int(max_iter):
        if abs(e[high - 1]) <= tol * (abs(d[high]) + abs(d[high - 1])):
            e[high - 1] = 0.0
            high -= 1
            continue
        block = symeig.tridiagonal_matrix(d[high - 1:high + 1], e[high - 1:high])
        mu = qralg.wilkinson_shift(block)
        x, z = d[0] - mu, e[0]
        for k in range(high):
            c, s = qr.givens_rotation(x, z)
            if k > 0:
                e[k - 1] = c * x + s * z
            t1, t2, t3 = d[k], e[k], d[k + 1]
            d[k] = c * c * t1 + 2 * c * s * t2 + s * s * t3
            e[k] = (c * c - s * s) * t2 + c * s * (t3 - t1)
            d[k + 1] = s * s * t1 - 2 * c * s * t2 + c * c * t3
            if k < high - 1:
                x, z = e[k], s * e[k + 1]
                e[k + 1] = c * e[k + 1]
        steps += 1
    return np.sort(d), steps
```

Measured against the dense version of lesson 37:

| `n` | tridiagonal steps | dense QR steps | max error |
|---|---|---|---|
| 10 | 21 | 21 | 5.33e-15 |
| 30 | 66 | 66 | 3.38e-14 |
| 60 | 127 | 121 | 1.14e-13 |

**The step counts are the same.** That is the correct and slightly surprising answer: the
tridiagonal version is not a better algorithm, it is the same algorithm on a cheaper
representation. Both take about 2 steps per eigenvalue, both use the Wilkinson shift, and both
converge cubically.

What changes is the cost per step: `O(n)` against `O(n^2)`, and the storage: `2n` numbers against
`n^2`. At `n = 60` that is a factor of 60 in work per step and a factor of 30 in memory, and both
factors grow with `n`. The small differences in step count, 127 against 121 at `n = 60`, come from
the different rounding paths, not from any difference in the method.

### 3.2 Divide and conquer eigenvectors, and the Lowner formula.

The naive formula takes `u_i` proportional to `(D - lambda_i I)^{-1} z` using the original `z`.
Each entry is `z_j / (d_j - lambda_i)`, and when `lambda_i` is very close to `d_j` that denominator
is a difference of nearly equal numbers computed with almost no relative accuracy. The resulting
vectors are not orthogonal.

Lowner's formula recovers a `z_hat` **consistent with the computed roots**:

```
z_hat_i^2  =  prod over j of (lambda_j - d_i)  /  prod over j != i of (d_j - d_i)
```

Every factor there is a difference of numbers that are not nearly equal, because the roots strictly
interlace the `d`'s by 38.2.4. Building the vectors from `z_hat` makes them orthogonal to working
precision even though the roots themselves still carry error.

```python
def lowner_vectors(d, z, roots):
    d = np.asarray(d, dtype=float)
    n = d.size
    z_hat = np.empty(n)
    for i in range(n):
        num = np.prod(roots - d[i])
        den = np.prod([(d[j] - d[i]) for j in range(n) if j != i])
        z_hat[i] = np.sqrt(abs(num / den)) * (1.0 if z[i] >= 0 else -1.0)
    V = np.empty((n, roots.size))
    for i, lam in enumerate(roots):
        w = z_hat / (d - lam)
        V[:, i] = w / np.linalg.norm(w)
    return V
```

Measured on `diag(d) + z z^T` with the `d_i` spaced by `gap`:

| gap between `d_i` | naive orthogonality | Lowner orthogonality | naive residual | Lowner residual |
|---|---|---|---|---|
| 1e-1 | 6.90e-15 | 2.68e-16 | 4.11e-15 | 2.69e-15 |
| 1e-3 | 1.23e-12 | 4.68e-16 | 7.60e-13 | 5.38e-13 |
| 1e-6 | 1.10e-9 | 6.43e-16 | 7.25e-10 | 5.12e-10 |
| 1e-9 | 2.34e-6 | 6.35e-16 | 1.43e-6 | 1.01e-6 |
| 1e-12 | 8.17e-4 | 2.70e-16 | 5.08e-4 | 3.60e-4 |

The naive orthogonality degrades by 11 orders of magnitude across the table, ending at 8e-4, which
is a completely unusable eigenvector matrix. **Lowner's is flat at 3e-16 to 6e-16 throughout.**

Note the two residual columns are nearly the same. That is the important nuance: Lowner does not
make the vectors more accurate as individual eigenvectors, because that accuracy is limited by the
error in the roots, which both share. It makes them **orthogonal**, which is a different property
and the one the divide and conquer recursion needs, since the vectors get multiplied together up
the tree and any loss of orthogonality compounds at every level.

### 3.3 One sided Jacobi gives the SVD directly.

Instead of rotating `A^T A` toward diagonal, rotate the **columns of `A`** until they are
orthogonal. The rotation that would annihilate the `(p, q)` entry of `A^T A` is computed from the
three inner products `a_p . a_p`, `a_q . a_q` and `a_p . a_q`, so `A^T A` is never formed. When the
columns are orthogonal, `A = U S` with `U` orthonormal and `S` the column norms, and the
accumulated rotations are `V`.

Measured with `nalib.svdcompute.one_sided_jacobi`:

| shape | `\|\|A - U S V^T\|\|` | `U` orthonormal | `V` orthonormal | values against LAPACK |
|---|---|---|---|---|
| 8 by 8 | 1.29e-14 | 1.18e-14 | 4.75e-15 | 4.44e-15 |
| 30 by 12 | 4.18e-14 | 1.13e-14 | 7.81e-15 | 9.77e-15 |
| 12 by 30 | 2.71e-14 | 5.48e-15 | 5.78e-15 | 7.99e-15 |
| 100 by 40 | 4.22e-13 | 6.34e-14 | 4.24e-14 | 7.11e-14 |

All four SVD properties hold at every shape. The wide case needs the matrix transposed first,
since the method works on columns and needs at least as many rows as columns; the two vector sets
then swap roles and nothing else changes.

This is lesson 42's method, and the reason it matters is exactly 38.2.5: because `A^T A` is never
formed, the small singular values keep their relative accuracy, and the guarantee is in terms of
`kappa(B)` rather than `kappa(A)`.

### Level 4, experimental

### 4.1 Cost of the four methods against `n`.

| method | fitted exponent | seconds at `n = 160` |
|---|---|---|
| Jacobi | 2.21 | 1.9644 |
| bisection | 1.85 | 0.4987 |
| divide and conquer | 1.41 | 0.4451 |
| LAPACK `eigvalsh` | 1.71 | 0.0015 |

The exponents are all below their theoretical values, which is an artifact worth naming: these are
Python implementations where the per step cost is dominated by interpreter overhead rather than
by flops, so the measured exponent reflects the number of numpy calls and not the arithmetic.
Jacobi at 2.21 is the closest to honest, since its `O(n^3)` per sweep is done in `O(n^2)` numpy
calls.

The number that matters is the last column. LAPACK is **1300 times faster than Jacobi** at
`n = 160`, and the gap grows. That is the reason Jacobi lost in the 1960s, and exercise 38.5.1
traces why it came back.

### 4.2 Jacobi accuracy tracks `kappa(B)`, not `kappa(A)`.

Two sweeps, with `A = D B D` and the reference computed by mpmath at 60 digits.

(a) `kappa(B)` held near 1.9, `kappa(A)` swept over 24 orders:

| `kappa(A)` | `kappa(B)` | Jacobi | LAPACK | `eps * kappa(B)` | `eps * kappa(A)` |
|---|---|---|---|---|---|
| 1.0e4 | 1.91 | 2.59e-15 | 2.84e-15 | 4.24e-16 | 2.32e-12 |
| 1.0e10 | 1.91 | 3.35e-15 | 1.14e-14 | 4.24e-16 | 2.32e-6 |
| 1.0e16 | 1.91 | 2.31e-15 | 2.12e-14 | 4.24e-16 | 2.32e0 |
| 1.0e22 | 1.91 | 2.28e-15 | 1.40e-13 | 4.24e-16 | 2.32e6 |
| 1.0e28 | 1.91 | 1.74e-15 | 1.25e-13 | 4.24e-16 | 2.32e12 |

Jacobi is **flat** at about 2e-15 while `kappa(A)` grows by 24 orders. The bound `eps kappa(A)`
exceeds 1 by the third row and is meaningless from there on, yet the answers stay perfect.

(b) `kappa(A)` held near 1e12 to 1e19, `kappa(B)` swept:

| `kappa(A)` | `kappa(B)` | Jacobi | LAPACK | `eps * kappa(B)` |
|---|---|---|---|---|
| 1.6e12 | 8.54 | 3.37e-15 | 1.18e-14 | 1.90e-15 |
| 2.8e13 | 6.73e2 | 1.10e-14 | 2.33e-13 | 1.49e-13 |
| 1.4e15 | 5.82e4 | 8.51e-13 | 2.24e-12 | 1.29e-11 |
| 1.0e17 | 5.31e6 | 2.59e-11 | 6.61e-11 | 1.18e-9 |
| 8.6e18 | 4.94e8 | 1.77e-9 | 8.22e-9 | 1.10e-7 |

Fitted: Jacobi error `~ kappa(B)^0.76`.

**That is the theorem confirmed.** The error follows `kappa(B)` and ignores `kappa(A)`. The fitted
exponent is 0.76 rather than 1 because the lowest rows sit near the machine precision floor, which
flattens the curve there. Dropping the flattest row raises it to 0.87, and over the top three rows
alone it is 0.85, so the trend is linear in `kappa(B)` with the floor pulling the fit down.

LAPACK is 3x to 8x worse than Jacobi throughout, and its error also follows `kappa(B)` rather than
`kappa(A)` on these matrices. That is worth stating plainly: **the guarantee separates them, the
observed behaviour on these matrices separates them by less than one order of magnitude.** A
guarantee is not an observed difference, and the value of the theorem is that it holds for every
matrix, including the ones where LAPACK would fail.

### 4.3 Jacobi sweeps against `n`, cyclic and greedy.

| `n` | cyclic sweeps | cyclic rotations | greedy sweeps | greedy rotations |
|---|---|---|---|---|
| 8 | 6 | 168 | 4 | 112 |
| 16 | 7 | 840 | 4 | 480 |
| 32 | 8 | 3968 | 4 | 1984 |
| 64 | 8 | 16128 | 5 | 10080 |

The sweep count grows extremely slowly, from 6 to 8 as `n` goes from 8 to 64, and for the greedy
ordering it is essentially constant at 4 or 5. That is the quadratic convergence of 38.2.2 showing
up: once the matrix is close to diagonal, each sweep squares the off diagonal norm, so the number
of sweeps needed is `log log` of the required accuracy, which is a constant for all practical
purposes.

Greedy uses about **40 percent fewer rotations** than cyclic. But the greedy ordering has to
**search** for the largest off diagonal entry before each rotation, which costs `O(n^2)`, while a
rotation itself costs `O(n)`. So greedy pays `O(n^2)` per rotation against cyclic's `O(n)`, and it
is a factor of `n` slower in practice despite doing fewer rotations. That is why every real
implementation uses the cyclic ordering, and why the theory for cyclic Jacobi, which is harder,
was worth developing.

### Level 5, advanced

### 5.1 Why Jacobi was abandoned and revived.

**Why it lost, in the 1960s.** The QR algorithm arrived with the tridiagonal reduction, and the
cost comparison was brutal. QR reduces to tridiagonal in `(4/3) n^3` flops and then finishes in
`O(n^2)`, so about `(4/3) n^3` total for eigenvalues. Jacobi does `O(n^3)` flops per sweep and
needs 6 to 10 sweeps, so about `6 n^3` to `10 n^3`. A factor of 5 to 8 on a machine measured in
kiloflops was decisive, and Jacobi disappeared from the libraries.

**What changed on the theory side.** Demmel and Veselic proved in 1992 what practitioners had
suspected: Jacobi is not merely as accurate as QR, it is **more** accurate, with a relative error
bound in `kappa(B)` rather than `kappa(A)`. That converted Jacobi from a slower alternative into
the only method with a guarantee for badly scaled matrices, which is a different product. Exercise
38.4.2 measures that bound holding across 24 orders of magnitude in `kappa(A)`.

**What changed on the hardware side.** Two things. Jacobi rotations on disjoint index pairs are
**independent**, so a sweep decomposes into `n/2` rotations that can run in parallel, and there
are orderings that keep every processor busy for the whole sweep. The QR algorithm's bulge chase
is inherently sequential. On a machine with thousands of cores the flop count stopped being the
figure of merit and the parallelism started being it.

The second is the memory hierarchy. Jacobi's rotations can be blocked so that a block of columns
stays in cache while many rotations are applied to it, giving level 3 BLAS performance. The
tridiagonal QR algorithm works on `2n` numbers with `O(n)` work per step and is completely memory
bound, so it achieves a small fraction of peak.

The result is that one sided Jacobi is now the method of choice for the SVD when accuracy matters
and for parallel machines generally, and it is in LAPACK as `dgesvj` and `dgejsv`. The measured
factor of 1300 in exercise 38.4.1 is a Python artifact; the real factor for tuned implementations
is closer to 2 to 5, and it goes the other way on enough cores.

### 5.2 The MRRR algorithm.

MRRR stands for Multiple Relatively Robust Representations, and it computes `k` eigenvectors of a
symmetric tridiagonal matrix in `O(nk)` total, with guaranteed orthogonality and **no**
reorthogonalization.

**The problem it solves.** Inverse iteration gives each eigenvector in `O(n)`, so `k` of them in
`O(nk)`, but by 36.5.1 the vectors for a cluster come out non-orthogonal and must be
reorthogonalized at a cost of `O(n c^2)` for a cluster of size `c`. For a spectrum that is one
large cluster this degrades to `O(n^3)`. Divide and conquer avoids that but costs `O(n^3)` in the
worst case anyway and always computes **all** the vectors, so asking for 10 of 100000 costs the
same as asking for all of them.

**Relatively robust representations.** A representation of a symmetric tridiagonal matrix is
relatively robust for a subset of eigenvalues if small relative changes to the stored quantities
cause only small relative changes to those eigenvalues. A `LDL^T` factorization is often relatively
robust where the tridiagonal entries themselves are not, because the factorization's entries
determine the small eigenvalues by products and quotients rather than by differences. The `dqds`
algorithm of exercise 42.3.2 is built on the same observation.

**The algorithm.** Compute the eigenvalues to high relative accuracy by bisection on an `LDL^T`
representation. Group them into clusters by relative gap. For each cluster, choose a **new shift**
`sigma` close to the cluster and form a new `LDL^T` factorization of `T - sigma I`. Within the
shifted representation, the cluster's eigenvalues are near zero, so their **relative** gaps are
large even though their absolute gaps were tiny. Recurse on the cluster with the new
representation until every eigenvalue is relatively isolated, then compute its eigenvector by a
twisted factorization in `O(n)`.

Orthogonality comes free because two eigenvectors computed from representations in which their
eigenvalues are relatively well separated are automatically orthogonal to working precision. There
is nothing to reorthogonalize.

The cost is `O(nk)` and the storage is `O(n)` plus the representation tree, whose depth is bounded
in practice by a small constant. It is LAPACK's `dstemr` and it is the default for the symmetric
eigenproblem when only some eigenpairs are wanted.

### 5.3 Clustered eigenvalues, and what actually degrades.

**The premise does not hold.** The exercise expects every method to degrade when eigenvalues
cluster. Measured on a 12 by 12 symmetric matrix with a cluster of 4 eigenvalues at spacing `gap`:

| gap | Jacobi orthogonality | `eigh` orthogonality | Jacobi vector error | `eigh` vector error | subspace error |
|---|---|---|---|---|---|
| 1e-2 | 4.69e-15 | 3.67e-15 | 1.88e-13 | 1.01e-13 | 1.22e-15 |
| 1e-6 | 6.23e-15 | 2.69e-15 | 1.64e-7 | 1.62e-9 | 1.82e-15 |
| 1e-10 | 6.38e-15 | 2.40e-15 | 7.64e-5 | 3.12e-6 | 1.51e-15 |
| 1e-14 | 5.87e-15 | 2.63e-15 | 2.09e-1 | 3.74e-2 | 1.79e-15 |
| 0 exactly | 5.68e-15 | 1.88e-15 | 1.37e0 | 1.29e0 | 1.37e-15 |

**Orthogonality of the basis holds at machine precision throughout**, for both methods, including
at an exactly repeated eigenvalue. It does not degrade at all. The same is true for the dense
Schur route, measured at 3.6e-15 to 4.6e-15 across the same gaps.

What does fall apart is the **individual eigenvector**, from 1.9e-13 at gap 1e-2 to 1.37 at gap
zero, meaning the computed vector is unrelated to the one it is compared against.

And that is the correct behaviour, not a failure. Inside an exact cluster **no individual
eigenvector is defined**: any orthonormal basis of the invariant subspace is a valid set of
eigenvectors, and the algorithm returns one of them chosen by rounding. There is no right answer
for it to get wrong.

The quantity that is well defined is the **subspace**, and the last column shows it is computed to
1.2e-15 to 1.8e-15 at every gap including zero. So the honest statement is:

- basis orthogonality: does not degrade, in any method
- the invariant subspace: does not degrade, in any method
- an individual eigenvector inside a cluster: degrades completely, in every method, because it
  is not a well posed quantity

The practical rule that follows is that a program which uses eigenvectors of a clustered spectrum
must use them in a way that is invariant to the choice of basis within each cluster, for example
by projecting onto the subspace. A program that uses a single vector from a cluster is computing
something that depends on rounding, and no algorithm can fix that.

---

## Lesson 39, Krylov Methods for Eigenvalues

### Level 1, conceptual

### 1.1 What stops the reductions of lessons 37 and 38 at `n = 10^6`?

Two things, and the second is worse.

Storage. A dense `10^6` by `10^6` matrix of doubles is 8 terabytes. The matrix itself is usually
sparse, with perhaps 10 nonzeros per row, so 80 megabytes, and it fits comfortably. But the
Hessenberg or tridiagonal reduction **fills it in**: the orthogonal similarity `Q^T A Q` destroys
sparsity, and the intermediate matrix is dense long before the reduction finishes.

Time. The reduction costs `O(n^3)`, which at `n = 10^6` is `10^18` flops. At a teraflop that is
30 years.

What a Krylov method does instead: it never forms any transformation of `A`. It builds the space
`span{v, Av, A^2 v, ..., A^{m-1} v}` for a modest `m`, typically 20 to 100, using one
matrix-vector product per step, and it solves the eigenvalue problem on the `m` by `m` projection.
The cost is `m` matrix-vector products, which for a sparse matrix is `O(m * nnz)`, plus an `m` by
`m` eigensolve, which is `O(m^3)` and negligible. Storage is the `m` basis vectors, `O(nm)`.

The matrix is only ever used through the operation `v -> A v`, so it need not be stored as a matrix
at all. It can be a function.

### 1.2 The Ritz residual is available without forming the Ritz vector. Why does that matter?

Because you need it for **every** Ritz value at **every** step, not once.

Forming one Ritz vector costs `y = V_m s`, which is `O(nm)`, one pass over the whole basis. Doing
that for all `m` Ritz values at every one of `m` steps costs `O(n m^3)`, which for `n = 10^6` and
`m = 100` is `10^12` operations, comparable to the whole rest of the computation.

The free formula `||A y - theta y|| = |h_{m+1,m}| |s_m|` costs one multiplication per Ritz value,
because `h_{m+1,m}` is already computed and `s_m` is the last entry of the small eigenvector, which
came from the `m` by `m` eigensolve. So checking all `m` residuals at every step costs `O(m^2)`
total, which is nothing.

The consequence is that a Krylov method can monitor convergence continuously and cheaply, decide
which Ritz values have converged, and form only those Ritz vectors, once, at the end. That is the
difference between a method that spends its time on the matrix and one that spends its time on
bookkeeping.

### 1.3 Lanczos without reorthogonalization returned an eigenvalue three times. Is the answer wrong?

No. The eigenvalue is correct, and the duplicates are **ghosts**, spurious copies produced when
rounding lets the Lanczos vectors drift back into a direction the iteration had already exhausted.

By Paige's theorem, that drift happens precisely when a Ritz value has **converged**. So a ghost
is a symptom of success, not of failure: the iteration found the eigenvalue, lost orthogonality
against it, and then found it again.

What to do about it, in increasing order of cost:

- **Nothing**, if you only want the eigenvalues. Take the distinct values and discard duplicates
  by proximity. The values are accurate.
- **The Cullum-Willoughby test**, exercise 39.3.3, which distinguishes ghosts from genuine
  multiple eigenvalues by comparing the spectrum of `T_m` with that of `T_m` with its first row
  and column deleted. It costs one extra small eigensolve and no extra storage.
- **Selective reorthogonalization**, which uses Paige's theorem to predict when orthogonality is
  about to be lost and reorthogonalizes only then, and only against the converged Ritz vectors.
  Costs a few percent.
- **Full reorthogonalization**, which orthogonalizes every new vector against the entire basis.
  Costs `O(nm^2)` and defeats the purpose for large `m`, but is simple and reliable, and is what
  a restarted method uses since its `m` stays small.

The one thing you must not do is conclude the run failed and start again with a different vector.
It will happen again, for the same reason.

### Level 2, mathematical

### 2.1 Prove the Ritz residual is orthogonal to the subspace.

Rayleigh-Ritz takes the subspace `K` with orthonormal basis `V`, forms `H = V^T A V`, solves
`H s = theta s`, and sets `y = V s`. The residual is `r = A y - theta y`.

Project onto the subspace:

```
V^T r = V^T A V s - theta V^T V s = H s - theta s = 0
```

using `V^T V = I`. So `r` is orthogonal to every vector in `K`. That is the defining property of
the Rayleigh-Ritz procedure, and it is called the Galerkin condition.

The consequence is that `theta` is the Rayleigh quotient of `y`:

```
y^T A y / y^T y = (V s)^T A (V s) / (V s)^T (V s) = s^T H s / s^T s = theta
```

so the Ritz value is exactly the Rayleigh quotient of the Ritz vector, and by 36.2.2 it is
therefore second order accurate in the error of `y` when `A` is symmetric. That is why Ritz values
converge faster than Ritz vectors, which exercise 39.4.2 sees directly.

### 2.2 Derive the free residual formula.

The Arnoldi relation is

```
A V_m = V_m H_m + h_{m+1,m} v_{m+1} e_m^T
```

where `V_m` has orthonormal columns, `H_m` is `m` by `m` upper Hessenberg, and `v_{m+1}` is
orthogonal to all of `V_m`. For a symmetric `A` this is the Lanczos relation and `H_m` is
tridiagonal.

Let `H_m s = theta s` with `||s|| = 1` and set `y = V_m s`. Then

```
A y = A V_m s = V_m H_m s + h_{m+1,m} v_{m+1} e_m^T s = theta V_m s + h_{m+1,m} s_m v_{m+1}
```

so

```
A y - theta y = h_{m+1,m} s_m v_{m+1}
```

and since `v_{m+1}` is a unit vector,

```
|| A y - theta y ||  =  |h_{m+1,m}| |s_m|
```

Both factors are already available: `h_{m+1,m}` from the last Arnoldi step, `s_m` as the last entry
of the small eigenvector.

**What changes without symmetry.** The formula itself is unchanged, since the derivation never used
symmetry. What changes is what the residual **means**. For a symmetric `A`, a small residual
implies the Ritz value is within `||r||` of a true eigenvalue, because the eigenvalues are
perfectly conditioned. For a non-symmetric `A`, Bauer-Fike gives only

```
distance to the spectrum  <=  kappa(V) ||r||
```

so a residual of 1e-14 with `kappa(V) = 1e8` says only that the eigenvalue is within 1e-6. The
residual is still the right thing to compute, but it must be multiplied by a condition number
before it becomes an error bound, and that condition number is exercise 35.4.3's individual number.

### 2.3 Prove Cauchy interlacing.

Let `A` be symmetric `n` by `n` with eigenvalues `lambda_1 >= ... >= lambda_n`, and let `V` be
`n` by `m` with orthonormal columns, `H = V^T A V` with eigenvalues `theta_1 >= ... >= theta_m`.

By Courant-Fischer applied to `H`,

```
theta_j = max over j-dimensional S in R^m of min over unit s in S of s^T H s
```

Each `j`-dimensional subspace `S` of `R^m` corresponds to the `j`-dimensional subspace `V S` of
`R^n`, and for `s` in `S` with `||s|| = 1`, the vector `y = V s` is a unit vector with
`y^T A y = s^T H s`. So the max-min over subspaces of `R^m` is a max-min over a **restricted
family** of subspaces of `R^n`, which can only be smaller than the max-min over all of them:

```
theta_j  <=  lambda_j
```

Applying the same argument to `-A` gives `theta_{m-j+1} >= lambda_{n-j+1}`, that is

```
lambda_{n-m+j}  <=  theta_j  <=  lambda_j
```

which is the interlacing. Every Ritz value lies inside the spectrum, and the `j`-th Ritz value is
squeezed between the `j`-th and the `(n-m+j)`-th eigenvalues.

The practical consequence is that Ritz values approach the extreme eigenvalues **from the inside**,
so `theta_1` is always a lower bound for `lambda_1` and `theta_m` is always an upper bound for
`lambda_n`. Those are rigorous bounds available at every step for free.

### 2.4 State Kaniel-Paige, and connect it to conjugate gradients.

**Kaniel-Paige.** For symmetric `A` with eigenvalues `lambda_1 > lambda_2 >= ... >= lambda_n` and
starting vector `v` with angle `phi_1` to the top eigenvector, the largest Ritz value after `m`
Lanczos steps satisfies

```
0  <=  lambda_1 - theta_1  <=  (lambda_1 - lambda_n) * ( tan(phi_1) / T_{m-1}(1 + 2 rho) )^2
```

where `rho = (lambda_1 - lambda_2)/(lambda_2 - lambda_n)` is the relative gap and `T_{m-1}` is the
Chebyshev polynomial of degree `m - 1`.

Since `T_k(1 + 2 rho)` grows like `(1 + 2 sqrt(rho))^k / 2` for small `rho`, the bound decays like

```
( 1 + 2 sqrt(rho) )^{-2(m-1)}
```

so the number of steps needed grows like `1 / sqrt(rho)`.

**The connection to conjugate gradients.** Lesson 24's CG bound is

```
||e_m||_A / ||e_0||_A  <=  2 ( (sqrt(kappa) - 1)/(sqrt(kappa) + 1) )^m
```

and it comes from exactly the same argument: the CG error after `m` steps is `p(A) e_0` minimised
over polynomials `p` of degree `m` with `p(0) = 1`, and the minimising polynomial is a scaled
Chebyshev polynomial on `[lambda_min, lambda_max]`.

Kaniel-Paige is the same minimisation with a different normalisation: minimise `p(A) v` over
polynomials of degree `m - 1` normalised at `lambda_1` instead of at 0. Both give a Chebyshev
polynomial, both give a `sqrt` of the relevant ratio in the exponent, and both are sharp for the
same reason: the Chebyshev polynomial is the one that is smallest on an interval given its value
at one outside point.

The `sqrt` in both is the single most important fact about Krylov methods. It is why they beat the
methods whose rate is the ratio itself, and it is why the two subjects, linear systems and
eigenvalues, have the same convergence theory.

### 2.5 State Paige's theorem, and explain why ghosts signal convergence.

**Paige's theorem.** In finite precision Lanczos without reorthogonalization, the loss of
orthogonality of the basis vector `v_{m+1}` against a converged Ritz vector `y_j` satisfies

```
| y_j^T v_{m+1} |   is about   eps ||A|| / ( |h_{m+1,m}| |s_m^{(j)}| )
```

that is, the loss of orthogonality is **inversely proportional to the residual** of that Ritz pair.
Equivalently the product of the two is about `eps ||A||`.

Measured on a 200 by 200 Laplacian:

| `m` | loss of orthogonality | smallest Ritz residual | product |
|---|---|---|---|
| 8 | 2.685e-15 | 7.661e-2 | 2.057e-16 |
| 16 | 5.076e-15 | 2.480e-2 | 1.259e-16 |
| 24 | 5.653e-15 | 1.476e-2 | 8.342e-17 |
| 32 | 8.441e-15 | 1.130e-2 | 9.542e-17 |
| 40 | 1.027e-14 | 1.086e-2 | 1.116e-16 |
| 48 | 1.027e-14 | 5.773e-3 | 5.930e-17 |
| 56 | 1.304e-14 | 3.750e-3 | 4.889e-17 |
| 64 | 1.562e-14 | 2.336e-3 | 3.649e-17 |
| 72 | 1.763e-14 | 1.871e-3 | 3.298e-17 |
| 80 | 2.059e-14 | 1.629e-3 | 3.354e-17 |

The product stays between 3.3e-17 and 2.1e-16 across the whole run, which is machine epsilon to
within a small factor, while the two factors individually move by an order of magnitude in opposite
directions. That is the quantitative form of the theorem and it is what exercise 39.4.2 asks for.

**Why ghosts signal convergence.** The theorem says orthogonality is lost against a Ritz vector
exactly when that Ritz vector's residual becomes small, that is, exactly when it converges. Once
orthogonality against `y_j` is lost, the iteration re-enters the direction `y_j`, and the Krylov
space starts rebuilding the component it had already used up. The next `T_m` therefore has a second
eigenvalue near `theta_j`, which is the ghost.

So a ghost cannot appear before the corresponding eigenvalue has converged. Seeing one is proof
that the answer is available, not evidence that it is not.

### Level 3, computational

### 3.1 Implicitly restarted Arnoldi.

Thick restart re-projects onto the wanted Ritz vectors and rebuilds. Sorensen's implicit restart
does something better: apply the `p` **unwanted** Ritz values as shifts in `p` steps of the QR
algorithm on the small matrix `T`. The accumulated `Q` filters the basis, and the leading `k`
columns of `V Q` span exactly the subspace a polynomial filter would have produced, at the cost of
`p` small QR steps and **no extra matrix-vector products at all**.

```python
def implicitly_restarted_lanczos(op, n, n_wanted, basis_size, tol=1e-10,
                                 max_restarts=2000, rng=None):
    g = np.random.default_rng() if rng is None else rng
    k, m = int(n_wanted), int(basis_size)
    v = g.standard_normal(n); v /= np.linalg.norm(v)
    V = np.zeros((n, m + 1)); V[:, 0] = v
    alpha = np.zeros(m); beta = np.zeros(m)
    matvecs = 0
    start = 0
    for restart in range(int(max_restarts)):
        for j in range(start, m):
            w = op(V[:, j]); matvecs += 1
            alpha[j] = float(V[:, j] @ w)
            w = w - alpha[j] * V[:, j] - (beta[j - 1] * V[:, j - 1] if j > 0 else 0.0)
            w = w - V[:, :j + 1] @ (V[:, :j + 1].T @ w)
            beta[j] = float(np.linalg.norm(w))
            if beta[j] < 1e-14:
                beta[j] = 0.0; V[:, j + 1] = 0.0
                break
            V[:, j + 1] = w / beta[j]
        T = np.diag(alpha) + np.diag(beta[:m - 1], 1) + np.diag(beta[:m - 1], -1)
        theta, S = np.linalg.eigh(T)
        order = np.argsort(theta)[::-1]
        theta, S = theta[order], S[:, order]
        resid = abs(beta[m - 1]) * np.abs(S[m - 1, :k])
        if np.all(resid <= tol * max(1.0, abs(theta[0]))):
            return {"values": theta[:k], "restarts": restart, "matvecs": matvecs,
                    "converged": True}
        Q_acc = np.eye(m)
        for mu in theta[k:]:                     # the UNWANTED values become the shifts
            Qs, Rs = np.linalg.qr(T - mu * np.eye(m))
            T = Rs @ Qs + mu * np.eye(m)
            Q_acc = Q_acc @ Qs
        V[:, :k] = V[:, :m] @ Q_acc[:, :k]
        rk = V[:, m] * (beta[m - 1] * Q_acc[m - 1, k - 1])
        alpha[:k] = np.diag(T)[:k]
        beta[:k - 1] = np.diag(T, -1)[:k - 1]
        nrk = float(np.linalg.norm(rk))
        beta[k - 1] = nrk
        V[:, k] = rk / nrk if nrk > 1e-300 else V[:, m]
        start = k
    return {"values": theta[:k], "restarts": int(max_restarts), "matvecs": matvecs,
            "converged": False}
```

Measured against the thick restart in `nalib.krylov_eig.restarted_arnoldi`, basis size 20:

| problem | wanted | thick matvecs | implicit matvecs | ratio | values agree to |
|---|---|---|---|---|---|
| geometric, `n = 60` | 1 | 78 | 39 | 2.00x | 7.11e-15 |
| geometric, `n = 60` | 4 | 111 | 52 | 2.13x | 1.24e-14 |
| linear, `n = 60` | 1 | 195 | 96 | 2.03x | 2.13e-14 |
| linear, `n = 60` | 4 | 219 | 100 | 2.19x | 3.55e-14 |
| clustered top, `n = 60` | 1 | 78 | 39 | 2.00x | 1.78e-15 |
| clustered top, `n = 60` | 4 | 219 | 100 | 2.19x | 5.33e-15 |

**The implicit restart uses almost exactly half the matrix-vector products**, consistently, across
all three spectra and both target counts, and reaches the same answers to 1e-14.

The factor of 2 has a clean explanation. Thick restart keeps `k` vectors and rebuilds from step
`k` to step `m`, costing `m - k` products per restart. Implicit restart keeps `k` vectors that
already encode the filter, and also rebuilds `m - k`, but the filtered starting vector is much
better, so it needs about half as many restarts. The measured ratio is the restart count ratio,
not a per restart saving.

This is the algorithm inside ARPACK, and by extension inside `scipy.sparse.linalg.eigsh`.

### 3.2 Shift and invert Lanczos for interior eigenvalues.

The operator `(A - sigma I)^{-1}` has eigenvalues `1/(lambda_i - sigma)`, so the eigenvalues of `A`
**nearest `sigma`** become the dominant ones and Lanczos finds them first. Each step costs a solve
instead of a product, so the factorization is done once and reused.

Measured on a 300 by 300 Laplacian, targeting the 150th of 300 eigenvalues, `2.01043714`:

| method | cost | nearest value found | error |
|---|---|---|---|
| plain Lanczos | 60 matrix-vector products | 2.06169683 | 5.13e-2 |
| shift and invert | 20 solves | 2.01043714 | 4.44e-16 |

Plain Lanczos with three times the work is off by 5e-2. Shift and invert is exact to the last bit.

**The trade.** Plain Lanczos converges to the **extreme** eigenvalues and reaches an interior one
only after it has resolved everything outside it, which for the middle of the spectrum means
essentially the whole problem. Shift and invert converts the interior target into an extreme one
and finds it immediately.

The cost comparison is the factorization. For this dense 300 by 300 case, one LU is `O(n^3)` and
each solve is `O(n^2)`, against `O(n^2)` per matrix-vector product, so 20 solves plus one
factorization costs about the same as 300 products, and plain Lanczos would need far more than
that to reach the same accuracy. For a **sparse** matrix the comparison depends entirely on
whether a sparse factorization is affordable: if the fill-in is modest, shift and invert wins by
orders of magnitude, and if the matrix is a 3-D discretization where the factorization fills in
catastrophically, it can be impossible and one falls back on an iterative solve inside the
iteration, which is the Jacobi-Davidson idea.

### 3.3 The Cullum-Willoughby ghost test.

A ghost is a spurious duplicate of an already converged Ritz value. Cullum and Willoughby noticed
that a ghost is **also** an eigenvalue of `T_m` with its first row and column removed, while a
genuine simple Ritz value is not. Comparing the two spectra and deleting the matches separates
them, with no reorthogonalization and no extra storage.

```python
def cullum_willoughby(alpha, beta, tol=1e-10):
    T = symeig.tridiagonal_matrix(alpha, beta)
    full = np.linalg.eigvalsh(T)
    trimmed = np.linalg.eigvalsh(T[1:, 1:]) if T.shape[0] > 1 else np.array([])
    keep = np.array([lam for lam in full
                     if trimmed.size == 0
                     or np.min(np.abs(trimmed - lam)) > tol * max(abs(lam), 1.0)])
    return keep, full, trimmed
```

Measured on a 40 by 40 Laplacian, run with no reorthogonalization for `m` steps:

| `m` | Ritz values | kept | removed | distinct true eigenvalues found | spurious before | spurious after |
|---|---|---|---|---|---|---|
| 40 | 40 | 40 | 0 | 40 | 0 | 0 |
| 60 | 60 | 40 | 20 | 40 | 20 | **0** |
| 100 | 100 | 0 | 100 | 40 | 60 | 0 |
| 200 | 200 | 0 | 200 | 40 | 160 | 0 |
| 400 | 400 | 0 | 400 | 40 | 360 | 0 |

At `m = 60` the test is exactly right: 20 of the 60 Ritz values are ghosts, it removes exactly
those 20, and the 40 survivors are the complete true spectrum with nothing spurious left.

At `m = 40` there is nothing to do, since the space has dimension 40 and no duplicates have
appeared yet.

**The limitation, stated honestly.** At `m = 100` and beyond the test deletes **everything**. Once
the iteration has run far past the dimension of the space, every eigenvalue has been found several
times over, so every Ritz value is also an eigenvalue of the trimmed matrix and nothing survives
the filter. The test works while ghosts are duplicates of a minority; it fails once the spectrum
is saturated with copies. In practice that is not a limitation, because nobody runs Lanczos to
ten times the dimension of the space, but it is worth knowing that the test has a range of
validity and it is not `m` unbounded.

### Level 4, experimental

### 4.1 Matvecs against the relative gap.

Restarted Arnoldi, basis size 15, one eigenvalue wanted, tolerance 1e-10:

| relative gap | restarts | matvecs | `1/sqrt(gap)` |
|---|---|---|---|
| 1e-1 | 3 | 87 | 3.2 |
| 1e-2 | 7 | 203 | 10.0 |
| 1e-3 | 23 | 667 | 31.6 |
| 1e-4 | 22 | 638 | 100.0 |

Fitted: `matvecs ~ gap^{-0.311}`.

**The measured exponent is 0.31, not the 0.5 that Kaniel-Paige predicts.** Two things are going on
and both are worth naming.

First, the count is not monotone: the gap of 1e-4 took **fewer** matvecs than 1e-3, 638 against
667. So the trend is noisy at this level and a fit through four points with that much scatter
cannot resolve 0.31 from 0.5 with confidence.

Second, and more substantively, Kaniel-Paige bounds **unrestarted** Lanczos from a single starting
vector. A restarted method applies a polynomial filter at every restart, and the composition of
many filters is a much higher degree polynomial than the basis size suggests. Restarting therefore
beats the single vector bound, which is exactly why restarting is done.

A cleaner way to read the same data is to form `matvecs * sqrt(gap)`, which would be constant if
the `1/sqrt(gap)` law held exactly:

| gap | 1e-1 | 1e-2 | 1e-3 | 1e-4 |
|---|---|---|---|---|
| `matvecs * sqrt(gap)` | 27.5 | 20.3 | 21.1 | 6.4 |

The first three are constant to within 30 percent, which is `1/sqrt(gap)` behaviour with a
constant near 20. The fourth breaks the pattern downward, meaning the restarted method did
**better** than the law at the smallest gap.

The honest summary: the matvec count grows as the gap shrinks, at a rate consistent with
`1/sqrt(gap)` over most of the range and better than it at the extreme, and the restart strategy
is what makes the difference. Kaniel-Paige is the right bound for the unrestarted method and a
conservative one for the restarted method.

### 4.2 Paige's theorem quantitatively.

Already tabulated in 39.2.5 above. The product of the loss of orthogonality and the smallest Ritz
residual stays between 3.3e-17 and 2.1e-16 over 80 Lanczos steps, while orthogonality decays from
2.7e-15 to 2.1e-14 and the residual falls from 7.7e-2 to 1.6e-3.

This is the quantitative statement, and it is stronger than the usual qualitative one. The usual
statement is "orthogonality is lost as Ritz values converge", which is a direction. The measurement
is "the product is machine epsilon", which is an equation and lets you **predict** the loss of
orthogonality from the residual, which you already have for free by 39.2.2. That prediction is
what selective reorthogonalization uses to decide when to act, without ever computing an inner
product to check.

### 4.3 The restart trade off.

Total matvecs to converge, tolerance 1e-10, spectrum `(1..40)^1.5`, against basis size:

| wanted | basis 4 | basis 6 | basis 10 | basis 15 | basis 25 | basis 40 |
|---|---|---|---|---|---|---|
| 1 | 252 | 187 | 152 | 116 | 98 | 79 |
| 3 | 1352 | 344 | 172 | 137 | 143 | 79 |
| 6 | - | - | 285 | 173 | 137 | 79 |

Counting matvecs alone, **bigger is always better**, and basis 40 is best in every row. But basis
40 equals the dimension of the problem, so there are no restarts at all and it is not a restarted
method.

The real trade off is not in the matvec count. It is:

- **Storage** is `O(n * basis)`. For `n = 10^6` a basis of 40 is 320 megabytes and a basis of 200
  is 1.6 gigabytes. That is often the binding constraint.
- **Orthogonalization cost** is `O(n * basis^2)` per restart cycle. For a cheap matrix-vector
  product, this dominates: at basis 200 the orthogonalization is 200 inner products per step
  against one product, so unless the matrix is expensive the method is spending its time on
  Gram-Schmidt.
- **The small eigensolve** is `O(basis^3)`, negligible until the basis reaches a few hundred.

What the table does show clearly is the **penalty for going too small**. For 3 wanted eigenvalues,
basis 4 costs 1352 matvecs against 137 at basis 15, a factor of 10. The rule that emerges is that
the basis must be comfortably larger than the number of wanted eigenvalues, by a factor of 3 or
more; below that the filter has too few unwanted Ritz values to work with and the restarts throw
away most of the progress. Above that, the matvec savings flatten out and the storage and
orthogonalization costs take over.

For the row `wanted = 1`, the sweet spot balancing all three is around basis 15 to 25, which is
why ARPACK's default is roughly `2 * wanted + 1` with a floor of 20.

### Level 5, advanced

### 5.1 Non-symmetric Krylov eigenvalues.

What changes when `A` is not symmetric:

**The recurrence lengthens.** Lanczos's three term recurrence exists because `V^T A V` is
symmetric tridiagonal. Without symmetry, `H_m = V^T A V` is full upper Hessenberg, so each new
vector must be orthogonalized against **all** previous ones. That is the Arnoldi process, and its
cost is `O(nm^2)` rather than `O(nm)`, and its storage is all `m` vectors rather than 3. This is
why restarting is not optional for non-symmetric problems.

**The Ritz values can be complex**, so the small eigensolve is a general one and the Ritz values
come in conjugate pairs for a real `A`.

**The interlacing is gone.** 39.2.3 used Courant-Fischer, which needs symmetry. Ritz values of a
non-symmetric matrix can lie outside the convex hull of the spectrum, and there is no free
inclusion bound.

**The residual no longer bounds the error.** By 39.2.2, a small residual gives
`distance <= kappa(V) ||r||`, so the error bound needs the conditioning. This is where lesson
35.4.3's individual condition number `1/|y^H x|` becomes essential: for a Ritz pair, the
corresponding quantity is computed from **both** the right and the left Ritz vectors, and it can be
enormous. A Ritz value with residual 1e-14 and condition number 1e10 is known only to 1e-4.

The practical response is to compute left Ritz vectors as well, so that the condition number is
available, which is what a two sided Arnoldi does, or to accept that the returned values are
backward stable rather than forward accurate and report the residual alongside.

```python
def arnoldi_eigen(op, n, m, rng=None):
    """Full Arnoldi with complete orthogonalization, returning Ritz values and their individual
    condition numbers from the left and right small eigenvectors."""
    g = np.random.default_rng() if rng is None else rng
    V = np.zeros((n, m + 1))
    H = np.zeros((m + 1, m))
    v = g.standard_normal(n); V[:, 0] = v / np.linalg.norm(v)
    for j in range(m):
        w = op(V[:, j])
        for i in range(j + 1):
            H[i, j] = V[:, i] @ w
            w = w - H[i, j] * V[:, i]
        for i in range(j + 1):                      # reorthogonalize once
            c = V[:, i] @ w
            H[i, j] += c
            w = w - c * V[:, i]
        H[j + 1, j] = np.linalg.norm(w)
        if H[j + 1, j] < 1e-14:
            m = j + 1
            break
        V[:, j + 1] = w / H[j + 1, j]
    Hm = H[:m, :m]
    theta, S = np.linalg.eig(Hm)
    _, Sl = np.linalg.eig(Hm.conj().T)
    cond = np.array([1.0 / max(abs(np.vdot(Sl[:, i], S[:, i])), 1e-300) for i in range(m)])
    resid = abs(H[m, m - 1]) * np.abs(S[m - 1, :]) if m < H.shape[0] else np.zeros(m)
    return {"values": theta, "residuals": resid, "condition": cond,
            "error_bound": resid * cond}
```

### 5.2 Block Lanczos.

A Krylov space built from **one** vector can contain at most one direction from any eigenspace.
The reason is immediate: on an eigenspace `E`, `A` acts as the scalar `lambda`, so
`span{v, Av, A^2 v, ...}` restricted to `E` is `span{P v, lambda P v, lambda^2 P v, ...}` where `P`
projects onto `E`, which is one dimensional. Therefore **an eigenvalue of multiplicity `p` can
never be found more than once by unblocked Lanczos**, no matter how many steps are taken.

Starting from `p` orthonormal vectors fixes exactly that: the space is
`span{V, AV, A^2 V, ...}` and its restriction to `E` is `span{P v_1, ..., P v_p}`, which is `p`
dimensional for a generic starting block.

```python
def block_lanczos(op, n, block, steps, rng=None):
    g = np.random.default_rng() if rng is None else rng
    p, s = int(block), int(steps)
    V, _ = np.linalg.qr(g.standard_normal((n, p)))
    basis = [V]
    for _ in range(s - 1):
        W = np.column_stack([op(basis[-1][:, j]) for j in range(p)])
        for _ in range(2):
            for B in basis:
                W = W - B @ (B.T @ W)
        Q, R = np.linalg.qr(W)
        keep = np.abs(np.diag(R)) > 1e-12 * max(1.0, float(np.abs(R).max()))
        if not keep.any():
            break
        basis.append(Q[:, keep])
    Vm = np.column_stack(basis)
    H = Vm.T @ np.column_stack([op(Vm[:, j]) for j in range(Vm.shape[1])])
    return np.linalg.eigvalsh(0.5 * (H + H.T)), Vm.shape[1]
```

Measured on an 80 by 80 operator with eigenvalue 5.0 repeated:

| multiplicity of 5.0 | block 1 finds | block 2 finds | block 4 finds |
|---|---|---|---|
| 1 | 1 | 1 | 1 |
| 2 | 2 | 2 | 2 |
| 3 | 2 | 2 | **3** |
| 4 | 2 | 2 | **4** |

Block 4 recovers the full multiplicity in every row. Blocks 1 and 2 saturate at 2 copies and never
find more, no matter that the basis reached size 34 and 38.

One honest caveat on the first two rows: in exact arithmetic block 1 should find only **one** copy
even at multiplicity 2, and it found two. That is rounding doing the work of a block, and it is the
same mechanism as the ghosts of 39.1.3: the reorthogonalization in this implementation is not
exact, so a second direction from the eigenspace leaks in. It is not reliable, as rows 3 and 4
show, and it is exactly the kind of accidental success that must not be depended on. The
theoretical statement stands: a block of size `p` is **required** to guarantee finding a
multiplicity of `p`.

The other advantage of blocking is practical: a block matrix-vector product `A V` with `V` having
`p` columns is a level 3 BLAS operation, which runs at a large multiple of the rate of `p`
separate level 2 products. So blocking often costs less wall clock time per unit of progress even
when it is not needed for multiplicity.

### 5.3 Why ARPACK stopped being the default.

Krylov methods build a subspace **sequentially**. Step `j + 1` needs the result of step `j`, so
the matrix-vector products cannot be overlapped, and the orthogonalization is a sequence of
inner products each of which is a global reduction. On a distributed machine, a global reduction
costs a latency that does not shrink with more processors, so the method's cost is bounded below
by `m` times the latency, no matter how many cores are available.

Two families of methods do something Krylov methods cannot:

**Randomised methods** replace the sequential build with a **single** block product `A Omega` for
a random `Omega` of `k + p` columns, followed by an orthogonalization and a small eigensolve.
Everything expensive is one level 3 BLAS operation with no sequential dependence at all, so it
parallelizes perfectly and streams. Lesson 43 develops this. The crossover is the spectrum: a
randomised method needs the spectrum to **decay**, because its error is governed by the tail
`sigma_{k+1}`, and it has no mechanism to resolve a cluster. Krylov methods resolve clusters and
have no decay requirement. So randomised wins for a low rank or fast decaying operator and loses
for a flat spectrum with a small gap.

**Contour integration methods**, FEAST and the Sakurai-Sugiura method, compute a spectral
projector directly by numerically integrating the resolvent around a contour in the complex plane:

```
P = (1 / 2 pi i) * integral over the contour of (z I - A)^{-1} dz
```

which projects onto the invariant subspace of the eigenvalues inside the contour. The integral is
approximated by a quadrature rule with a modest number of nodes, typically 8 to 16, and each node
is an **independent linear solve** `(z_j I - A)^{-1} Y`. Those solves have no dependence on each
other, so they run on separate machines simultaneously. Applying `P` to a random block and doing
Rayleigh-Ritz gives every eigenvalue inside the contour.

What that buys: perfect parallelism across quadrature nodes, the ability to target an arbitrary
region of the complex plane rather than only extremes, and natural handling of multiplicity since
the projector does not care. What it costs: a linear solve per node per iteration, which is the
same requirement as shift and invert and can be prohibitive, and a quadrature error that must be
controlled by adding nodes.

The crossover is roughly this. If you want a few extreme eigenvalues and a factorization is
unaffordable, Krylov is still the right answer and ARPACK is still what you use. If you want all
the eigenvalues in a window, or you have many cores and a factorization is affordable, contour
integration wins. If your operator has a rapidly decaying spectrum and you want a low rank
approximation rather than eigenvalues as such, randomised methods win and win by a lot.

---

## Lesson 40, The Generalized Eigenvalue Problem

### Level 1, conceptual

### 1.1 `B^{-1} A x = lambda x` has the right eigenvalues. Name three things it loses.

**One, symmetry.** If `A` and `B` are both symmetric, `B^{-1} A` is in general **not** symmetric,
because the product of two symmetric matrices is symmetric only when they commute. So the problem
is handed to a general eigensolver, losing the real eigenvalues, the orthogonal eigenvectors, the
perfect conditioning, and the factor of two in cost that symmetry buys.

**Two, accuracy.** Forming `B^{-1} A` costs a factorization and a solve, and the result carries a
relative error of about `eps kappa(B)`. Exercise 40.4.1 measures that this is not a pessimistic
bound but the observed behaviour.

**Three, infinite eigenvalues.** If `B` is singular the product does not exist at all, and if `B`
is nearly singular the product has enormous entries that are mostly noise. The pencil may have
perfectly well defined finite eigenvalues alongside infinite ones, and `B^{-1} A` cannot represent
that.

**The one that costs most is the first.** The accuracy loss is real but bounded and often
acceptable, and the singular case can be detected. Losing symmetry changes the problem class: it
turns a well conditioned problem with a guaranteed real answer into a general one with no such
guarantee, and it doubles or quadruples the cost. The Cholesky reduction of 40.2.1 exists
precisely to avoid this.

### 1.2 A pencil with singular `B` has fewer than `n` finite eigenvalues. Where did the others go?

They went to infinity, and that is a meaningful statement rather than a failure.

Write the eigenvalue problem as `beta A x = alpha B x`, so `lambda = alpha / beta`. If `B x = 0`
and `A x != 0`, then `beta` must be zero and `alpha` is nonzero, so the pair `(alpha, 0)`
represents an eigenvalue at infinity. Physically this is a constraint: in a mechanical system with
a massless degree of freedom, that mode has infinite frequency, meaning it responds
instantaneously and is not a vibration at all.

**Why a single number cannot report them.** `lambda = alpha / beta` is undefined when `beta = 0`,
and any floating point value returned for it is a lie. Returning `inf` conflates a genuine infinite
eigenvalue with an overflow, and returning a huge finite number loses the distinction entirely.

The fix is to return the **pair** `(alpha, beta)` and let the caller form the ratio when it is safe.
That is what LAPACK's `dggev` does, returning `alphar`, `alphai` and `beta` separately, and it is
why `nalib.geneig.PencilResult` carries both. The pair also handles the genuinely indeterminate
case `alpha = beta = 0`, which signals a **singular pencil** where every `lambda` is an eigenvalue,
and no single number could ever express that.

### 1.3 Why does QZ need two unitary matrices where QR needs one?

Counting. The QR algorithm computes `Q^H A Q`, one matrix `Q` with about `n^2/2` free parameters,
and needs to reduce `A` to triangular form, which means killing about `n^2/2` subdiagonal entries.
The counts match.

QZ must reduce **two** matrices, `A` and `B`, to triangular form simultaneously. That is about
`n^2` entries to kill, so one unitary matrix does not have enough parameters. Using two, as
`Q^H A Z` and `Q^H B Z`, gives about `n^2` parameters and the counts match again. The equivalence
transformation `(A, B) -> (Q^H A Z, Q^H B Z)` preserves the eigenvalues of the pencil because
`det(Q^H A Z - lambda Q^H B Z) = det(Q^H) det(A - lambda B) det(Z)`, which vanishes exactly when
`det(A - lambda B)` does.

### Level 2, mathematical

### 2.1 Prove that `L^{-1} A L^{-T}` is symmetric with the same eigenvalues.

Let `B = L L^T` be the Cholesky factorization, which requires `B` positive definite. Set
`C = L^{-1} A L^{-T}`. Then

```
C^T = (L^{-1} A L^{-T})^T = L^{-1} A^T L^{-T} = L^{-1} A L^{-T} = C
```

using only `A^T = A`, so `C` is symmetric.

For the eigenvalues, `A x = lambda B x` with `B = L L^T` gives

```
A x = lambda L L^T x
```

Substitute `y = L^T x`, so `x = L^{-T} y`, and multiply on the left by `L^{-1}`:

```
L^{-1} A L^{-T} y = lambda y,   that is   C y = lambda y
```

So the pencil's eigenvalues are exactly `C`'s, and the eigenvectors are related by `x = L^{-T} y`.

**Where positive definiteness is used**, in two distinct places. First, the Cholesky factorization
`B = L L^T` with real `L` exists **only** for a positive definite `B`; for an indefinite `B` there
is no such factorization over the reals. Second, `L` must be **invertible** for the substitution
`y = L^T x` to be a bijection, which needs `B` nonsingular. Positive definiteness gives both at
once.

The conditioning of the reduction is governed by `kappa(L) = sqrt(kappa(B))`, and it enters twice,
once on each side, so the overall sensitivity is about `kappa(B)`. But crucially the errors are
**symmetric**, so they perturb the eigenvalues of a symmetric matrix, which are perfectly
conditioned. That is why exercise 40.4.1 measures the Cholesky route to be flat in `kappa(B)`
while the naive route is linear in it.

### 2.2 Prove the eigenvectors of a symmetric definite pencil are `B`-orthogonal.

Let `A x_i = lambda_i B x_i` and `A x_j = lambda_j B x_j` with `lambda_i != lambda_j`. Then

```
x_j^T A x_i = lambda_i x_j^T B x_i
x_i^T A x_j = lambda_j x_i^T B x_j
```

Transposing the second and using the symmetry of both `A` and `B`,

```
x_j^T A x_i = lambda_j x_j^T B x_i
```

Subtracting, `(lambda_i - lambda_j) x_j^T B x_i = 0`, and since the eigenvalues differ,

```
x_j^T B x_i = 0
```

which is `B`-orthogonality. Normalising so that `x_i^T B x_i = 1` gives `X^T B X = I` and
`X^T A X = diag(lambda)`.

**Why it decouples the modes.** For `M u'' + K u = 0` with `M` and `K` symmetric and `M` positive
definite, substitute `u = X q` where `X` holds the eigenvectors of `K x = lambda M x`:

```
M X q'' + K X q = 0
```

Multiply on the left by `X^T`:

```
(X^T M X) q'' + (X^T K X) q = 0,   that is   q'' + diag(lambda) q = 0
```

The system is now `n` **independent** scalar equations `q_i'' + lambda_i q_i = 0`, each an
oscillator of frequency `sqrt(lambda_i)`. That is the entire content of modal analysis: `B`
orthogonality is what turns a coupled system into uncoupled ones, and it is the reason engineers
want the generalized problem solved as a pencil rather than converted.

### 2.3 Show that a common null vector makes the pencil singular.

Suppose `x != 0` with `A x = 0` and `B x = 0`. Then for **every** scalar `lambda`,

```
(A - lambda B) x = 0 - lambda * 0 = 0
```

so `A - lambda B` is singular for every `lambda`, and `det(A - lambda B)` is identically zero as a
polynomial. Every complex number is an eigenvalue. The pencil is called **singular**.

**How this differs from merely having infinite eigenvalues.** A pencil with `B` singular but no
common null vector is **regular**: `det(A - lambda B)` is a nonzero polynomial of degree less than
`n`, and the deficiency in degree is exactly the number of infinite eigenvalues. The finite
eigenvalues are the roots of that polynomial, they are well defined, and QZ computes them.

For a singular pencil there is nothing to compute. The characteristic polynomial is the zero
polynomial, the eigenvalues are not a finite set, and any algorithm that returns `n` numbers is
returning noise. Worse, singularity is not numerically detectable in general, because an arbitrarily
small perturbation makes a singular pencil regular, with `n` eigenvalues that depend entirely on
the perturbation. This is the pencil analogue of an ill posed problem, and the correct response is
to detect the common null space with an SVD of the stacked matrix `[A; B]` and reformulate, not to
call an eigensolver.

### 2.4 Prove the existence of the generalized Schur form.

**Claim.** For any square `A` and `B` there exist unitary `Q` and `Z` with `Q^H A Z = S` and
`Q^H B Z = T` both upper triangular.

Proof by induction on `n`. For `n = 1` there is nothing to prove.

For the step, `det(A - lambda B)` is a polynomial in `lambda` that is either identically zero, in
which case the pencil is singular and we may take any eigenvalue, or has a root. In either case
choose `(alpha, beta)` not both zero with `det(beta A - alpha B) = 0`, which exists because
`det(beta A - alpha B)` is a homogeneous polynomial in `(alpha, beta)` and has a nontrivial zero
over the complex numbers. Let `z` be a unit vector with `(beta A - alpha B) z = 0`.

Set `w = A z` and `u = B z`. These satisfy `beta w = alpha u`, so `w` and `u` are parallel; let `q`
be a unit vector spanning their common direction, or any unit vector if both are zero. Extend `z`
to a unitary `Z_1` with `z` as first column, and `q` to a unitary `Q_1` with `q` as first column.
Then both `Q_1^H A Z_1` and `Q_1^H B Z_1` have first column `(*, 0, ..., 0)^T`, because
`Q_1^H A z = Q_1^H w` is a multiple of `Q_1^H q = e_1`, and likewise for `B`.

So both matrices have the block form with a zero block below the first entry, and induction on the
trailing `(n-1)` by `(n-1)` pencil completes the proof.

**Why one unitary matrix is not enough.** With `Q^H A Q` and `Q^H B Q` there are only about `n^2/2`
free parameters and about `n^2` entries to annihilate, as counted in 40.1.3. Concretely, a single
similarity that triangularizes `A` will in general leave `B` full, and no ordering of the choices
fixes that. The two sided form is not a convenience, it is a necessity forced by parameter
counting.

### 2.5 Perturbation theory in the chordal metric.

The ordinary distance `|lambda - mu|` is inadequate for a pencil for one reason: the eigenvalues
live on the **Riemann sphere**, not on the complex plane, and infinity is an ordinary point there.
Two eigenvalues at 1e15 and 2e15 are far apart in the ordinary metric and nearly the same point on
the sphere, and an eigenvalue moving from 1e15 to infinity is an infinite ordinary distance and a
tiny movement on the sphere.

The **chordal metric** measures the straight line distance between the two points on the sphere
under stereographic projection. For eigenvalues represented as pairs `(alpha_1, beta_1)` and
`(alpha_2, beta_2)`,

```
chord( (a_1, b_1), (a_2, b_2) )  =  |a_1 b_2 - a_2 b_1| / ( sqrt(|a_1|^2 + |b_1|^2) sqrt(|a_2|^2 + |b_2|^2) )
```

which is symmetric, bounded by 1, and treats infinity as an ordinary point.

**The theorem.** For a symmetric definite pencil `(A, B)` with Crawford number `c(A, B)`, and
perturbations `E` and `F` with `sqrt(||E||^2 + ||F||^2) < c(A, B)`, every eigenvalue of the
perturbed pencil satisfies

```
chord(lambda_perturbed, lambda_original)  <=  sqrt(||E||^2 + ||F||^2) / c(A, B)
```

So the Crawford number is the condition number of a definite pencil in the chordal metric, exactly
as `1/|y^H x|` is for a standard eigenvalue. The pencil is well conditioned when `c(A, B)` is large
relative to the perturbations, and the bound degrades as the pencil approaches indefiniteness,
which is where `c(A, B)` goes to zero. Exercise 40.5.1 computes `c(A, B)` and shows it detecting
definiteness that the naive test misses.

### Level 3, computational

### 3.1 The QZ algorithm.

**Step one, Hessenberg-triangular reduction.** Make `B` upper triangular by a QR factorization,
then drive `A` to Hessenberg with Givens rotations from the left, repairing the fill each rotation
creates in `B` with a second rotation from the right.

```python
def hessenberg_triangular(A, B):
    A = np.array(A, dtype=float, copy=True)
    B = np.array(B, dtype=float, copy=True)
    n = A.shape[0]
    Q, B = np.linalg.qr(B)
    A = Q.T @ A
    Qa, Za = Q.copy(), np.eye(n)
    for j in range(n - 2):
        for i in range(n - 1, j + 1, -1):
            c, s = _giv(A[i - 1, j], A[i, j])
            G = np.array([[c, s], [-s, c]])
            A[i - 1:i + 1, :] = G @ A[i - 1:i + 1, :]
            B[i - 1:i + 1, :] = G @ B[i - 1:i + 1, :]
            Qa[:, i - 1:i + 1] = Qa[:, i - 1:i + 1] @ G.T
            c, s = _giv(-B[i, i], B[i, i - 1])        # repair the fill in B
            G = np.array([[c, s], [-s, c]])
            A[:, i - 1:i + 1] = A[:, i - 1:i + 1] @ G.T
            B[:, i - 1:i + 1] = B[:, i - 1:i + 1] @ G.T
            Za[:, i - 1:i + 1] = Za[:, i - 1:i + 1] @ G.T
    return np.triu(A, -1), np.triu(B), Qa, Za
```

Measured:

| `n` | `\|\|Q^T A Z - H\|\|` | `\|\|Q^T B Z - T\|\|` | `H` below subdiagonal | `T` below diagonal | `Q`, `Z` orthogonal |
|---|---|---|---|---|---|
| 6 | 1.36e-15 | 7.48e-15 | 0 | 0 | 1.17e-15 |
| 10 | 3.56e-15 | 1.21e-14 | 0 | 0 | 1.64e-15 |
| 20 | 1.08e-14 | 5.48e-14 | 0 | 0 | 3.04e-15 |
| 40 | 3.11e-14 | 2.22e-13 | 0 | 0 | 6.19e-15 |
| 80 | 8.63e-14 | 8.47e-13 | 0 | 0 | 1.38e-14 |

The reduction is exact to rounding and the structure is exact to the bit.

**Step two, the eigenvalues.** With the pair reduced, the eigenvalues follow. Compared against the
library:

| `n` | `kappa(T)` | spectrum matches library to |
|---|---|---|
| 6 | 1.63 | 3.21e-14 |
| 10 | 2.15 | 9.30e-15 |
| 20 | 1.80 | 6.21e-15 |
| 40 | 1.52 | 1.08e-14 |
| 80 | 1.37 | 5.90e-15 |

**Where the shortcut breaks, and why the real QZ iteration exists.** The implementation above
finishes by running the QR algorithm on `H T^{-1}`, which is legitimate only when `T` is well
conditioned. Measured with `kappa(B)` swept:

| `kappa(B)` | error of the `H T^{-1}` route |
|---|---|
| 1e2 | 8.20e-12 |
| 1e6 | 1.22e-6 |
| 1e10 | 2.66e2 |
| 1e14 | 2.48e10 |

The error tracks `eps kappa(B)` exactly, and by `kappa(B) = 1e10` the answers are meaningless. That
is the whole reason the genuine QZ iteration exists: it performs the shifted step **implicitly on
the pair**, alternating a rotation on `H` with a repairing rotation on `T`, and never forms
`T^{-1}` or any inverse at all. The bulge chase carries the shift through both matrices
simultaneously, so the accuracy is governed by the orthogonal transformations and not by
`kappa(B)`. Implementing the bulge chase correctly is intricate, which is why LAPACK's `dhgeqz`
runs to about 700 lines, and getting it wrong produces an iteration that does not converge.

### 3.2 Shift and invert Lanczos for a pencil.

Solve with `A - sigma B` and `B`-orthogonalize the basis. The operator is `(A - sigma B)^{-1} B`,
whose eigenvalues are `1/(lambda - sigma)`, so the frequencies nearest `sigma` become dominant.
`B`-orthogonality is what keeps the projected pair symmetric and the Ritz values real.

```python
def shift_invert_pencil(K, M, sigma, m, rng=None):
    g = np.random.default_rng() if rng is None else rng
    n = K.shape[0]
    lu = sla.lu_factor(K - sigma * M)
    v = g.standard_normal(n)
    v /= np.sqrt(float(v @ (M @ v)))
    V = [v]
    for _ in range(int(m) - 1):
        w = sla.lu_solve(lu, M @ V[-1])
        for _ in range(2):                       # M-orthogonalize, twice for safety
            for u in V:
                w = w - u * float(u @ (M @ w))
        nb = float(np.sqrt(max(w @ (M @ w), 0.0)))
        if nb < 1e-12:
            break
        V.append(w / nb)
    Vm = np.column_stack(V)
    Km, Mm = Vm.T @ K @ Vm, Vm.T @ M @ Vm
    return sla.eigh(0.5 * (Km + Km.T), 0.5 * (Mm + Mm.T), eigvals_only=True), Vm.shape[1]
```

Measured on a 400 element string with density `1 + 3x`, targeting the 200th of 400 frequencies,
`0.80291772`, with `sigma` set 2 percent below it:

| method | cost | error |
|---|---|---|
| shift and invert | 10 solves | 6.004e-6 |
| shift and invert | 20 solves | 2.343e-14 |
| shift and invert | 40 solves | 5.551e-16 |
| plain Lanczos on `M^{-1} K` | 40 matvecs | 3.754e-2 |

Twenty solves reach 2e-14 on an interior frequency that plain Lanczos with twice the steps misses
by 4e-2. This is the same result as 39.3.2 in the pencil setting, and the `B`-orthogonalization is
the only structural addition.

For a real structure the matrix `K - sigma M` is sparse and banded, so the factorization is cheap
and this is the standard method for modal analysis in a frequency band. Sweeping `sigma` across the
band and collecting the modes found near each shift is how commercial finite element packages
compute mode shapes.

### 3.3 The Crawford number.

```python
def crawford_number(A, B, n_samples=400_000, rng=None):
    """min over unit x of hypot(x^T A x, x^T B x). Positive means the pencil is DEFINITE:
    some combination cos(t) A + sin(t) B is positive definite, even when neither A nor B is."""
    gen = np.random.default_rng() if rng is None else rng
    X = gen.standard_normal((A.shape[0], int(n_samples)))
    X /= np.linalg.norm(X, axis=0)
    a = np.einsum("ij,ij->j", X, A @ X)
    b = np.einsum("ij,ij->j", X, B @ X)
    return float(np.min(np.hypot(a, b)))
```

Sampling gives an upper bound on `c(A, B)`, which is the useful direction: if the sampled value is
already small the pencil is close to indefinite. For a certified lower bound one bisects on the
angle, testing whether `cos(t) A + sin(t) B` is positive definite by attempting a Cholesky
factorization, which is what the sweep in 40.5.1 does.

Measured, a definite pencil where **neither** matrix is definite:

```
eigenvalues of A : [-1.5 -0.5  0.5  1.5  2.5  3.5]
eigenvalues of B : [-3.5 -2.5 -1.5 -0.5  0.5  1.5]
B positive definite : False    A positive definite : False
naive test reports definite = False
Crawford number  c(A, B) = 1.4142 > 0, so the pencil IS definite
QZ eigenvalues are real: max |imag| = 0.00e+00
```

And for contrast, a random `A` with the same `B`:

```
Crawford number 0.0032, max |imag| of the eigenvalues 1.340
```

The Crawford number separates the two cases cleanly where the test on `B` alone gets the first one
wrong. That is exercise 40.5.1's subject and it is developed there.

### Level 4, experimental

### 4.1 The two reductions against `kappa(B)`.

| `kappa(B)` | Cholesky route | naive `B^{-1}A` route | `eps * kappa(B)` | naive / bound |
|---|---|---|---|---|
| 1e2 | 1.11e-15 | 1.89e-15 | 2.22e-14 | 0.085 |
| 1e4 | 1.02e-15 | 3.15e-14 | 2.22e-12 | 0.014 |
| 1e6 | 4.82e-16 | 3.98e-12 | 2.22e-10 | 0.018 |
| 1e8 | 2.45e-15 | 9.79e-10 | 2.22e-8 | 0.044 |
| 1e10 | 1.46e-14 | 8.45e-9 | 2.22e-6 | 0.004 |
| 1e12 | 1.60e-14 | 6.59e-6 | 2.22e-4 | 0.030 |
| 1e14 | 1.69e-14 | 1.35e-4 | 2.22e-2 | 0.006 |

Fitted: Cholesky error `~ kappa(B)^0.13`, naive error `~ kappa(B)^0.94`.

**The naive route is linear in `kappa(B)`, exactly as `u kappa(B)` predicts.** The exponent 0.94
against the predicted 1.00 is as close as a seven point fit over twelve orders gets.

**The Cholesky route is nearly flat**, exponent 0.13, growing only from 1e-15 to 1.7e-14 across the
whole range, which is a factor of 17 while the naive route grows by a factor of 7e10. The reason is
40.2.1: the Cholesky reduction produces a **symmetric** matrix, whose eigenvalues are perfectly
conditioned, so the errors introduced by the reduction perturb a well conditioned problem. The
naive route produces a non-symmetric matrix whose eigenvalue conditioning is governed by
`kappa(B)`.

The bound is loose by a factor of 11 to 250, which is normal for a worst case bound applied to a
random matrix: the bound must hold for the worst perturbation direction and a random one is not
worst.

### 4.2 QZ against the Cholesky route.

| `n` | Cholesky | QZ | ratio | LAPACK `eigh` |
|---|---|---|---|---|
| 50 | 0.0012s | 0.0031s | 2.6x | 0.0003s |
| 100 | 0.0037s | 0.0287s | 7.7x | 0.0019s |
| 200 | 0.0123s | 0.1559s | 12.7x | 0.0077s |
| 400 | 0.0641s | 1.1271s | 17.6x | 0.0435s |

**QZ costs 2.6x to 17.6x more, and the ratio grows with `n`.** The growth is the important part: it
means the two have different constants on the same `O(n^3)`, and the gap widens rather than
settling. The published flop counts are about `66 n^3` for QZ with both `Q` and `Z` accumulated
against about `(4/3) n^3` for a Cholesky factorization plus a symmetric reduction and solve, a
ratio near 50, and the measured 17.6x at `n = 400` is heading in that direction as the LAPACK
constants stop dominating.

So the rule is clear: **use the Cholesky route whenever it applies**, meaning whenever `B` is
symmetric positive definite and not too ill conditioned. Use QZ when `B` is indefinite, singular,
or so ill conditioned that 40.4.1's `eps kappa(B)` is unacceptable. QZ is the general method and
you pay for the generality.

### 4.3 A graded structure against the analytic answer.

The discrete problem is `K x = lambda M x` with `K = tridiag(-1, 2, -1)` and `M = diag(rho)`.
Multiplying by `(n+1)^2` gives the continuum problem `-u'' = lambda rho(x) u` on `[0, 1]` with
fixed ends, whose lowest frequency by the WKB estimate is `(pi / integral of sqrt(rho))^2`.

Measured at `n = 400`:

| grading | computed `lambda_1` | WKB estimate | ratio | mode peak at `x` |
|---|---|---|---|---|
| uniform, `rho = 1` | 9.8696 | 9.8696 | 1.0000 | 0.501 |
| linear, `rho = 1 + 3x` | 3.8883 | 4.0788 | 0.9533 | 0.544 |
| steep, `rho = exp(4x)` | 0.8901 | 0.9671 | 0.9204 | 0.636 |
| two blocks, 1 then 9 | 1.7374 | 2.4674 | 0.7042 | 0.603 |

For uniform density the agreement is **exact to four decimals**, since `pi^2 = 9.8696` and the WKB
estimate is the true answer there.

As the grading steepens the WKB estimate degrades, from 4.7 percent error for the linear density to
**30 percent** for the discontinuous two block case. That is the correct behaviour and it identifies
the assumption: WKB assumes the density varies **slowly** compared with the wavelength, and a jump
discontinuity violates that maximally. The estimate is an approximation whose error is controlled
by the smoothness of `rho`, and the table measures exactly that.

The mode shape column is the physical result. The peak of the fundamental mode moves from the
centre, `x = 0.501`, toward the **heavy** end as the grading steepens, reaching `x = 0.636` for the
exponential density. That is what a heavier region does: it moves more slowly, so more of the
motion is concentrated there, and the frequency drops accordingly, from 9.87 to 0.89.

### Level 5, advanced

### 5.1 Definiteness is not a property of `B` alone.

**The correct definition.** A Hermitian pencil `(A, B)` is **definite** when

```
c(A, B) = min over unit x of | x^H A x + i x^H B x |  >  0
```

Equivalently, some combination `cos(t) A + sin(t) B` is positive definite. The quantity `c(A, B)`
is the **Crawford number**, and it is the correct condition number for the pencil by 40.2.5.

The naive test, "is `B` positive definite", is a **sufficient** condition, not a necessary one. It
corresponds to checking only `t = pi/2`.

**A pencil the naive test rejects and the correct test accepts.** Take `d = (1, 2, 3, 4, 5, 6)` and
a random orthogonal `Q`, and set

```
A = Q diag(d - 2.5) Q^T,      B = Q diag(d - 4.5) Q^T
```

Both shifts sit inside the range of `d`, so both matrices have mixed signs. Measured:

```
eigenvalues of A : [-1.5 -0.5  0.5  1.5  2.5  3.5]
eigenvalues of B : [-3.5 -2.5 -1.5 -0.5  0.5  1.5]
B positive definite : False      A positive definite : False
the naive test reports definite = False, so the Cholesky route is unavailable

Crawford number c(A, B) = 1.4142 > 0, so the pencil IS definite
the definite combinations form an arc of width 0.760 rad around t = -0.785
best member: cos(-0.785) A + sin(-0.785) B has smallest eigenvalue +1.4142
QZ eigenvalues are real: max |imag| = 0.00e+00
they match (d_i - 2.5)/(d_i - 4.5) to 5.33e-15
```

The construction makes the reason transparent: `A - B = (4.5 - 2.5) I = 2I` is positive definite,
so the combination at `t = -pi/4`, which is `(A - B)/sqrt(2) = sqrt(2) I`, is definite with
smallest eigenvalue exactly `sqrt(2) = 1.4142`. The bisection over `t` finds an arc of definite
combinations 0.76 radians wide centred on that point.

And the eigenvalues behave as a definite pencil must: all real, matching the exact ratios
`(d_i - 2.5)/(d_i - 4.5)` to 5.3e-15.

**For contrast**, a random symmetric `A` with the same `B`:

```
Crawford number 0.0032 (essentially zero), max |imag| of the eigenvalues 1.340
```

The eigenvalues are genuinely complex, which a definite pencil's cannot be.

**What this changes in practice.** If the naive test fails, you are not obliged to fall back on QZ
and pay the 17.6x of 40.4.2. Compute the Crawford number, find the definite combination
`C = cos(t) A + sin(t) B`, and reduce the equivalent pencil `(A, C)` or `(C, B)` by Cholesky, then
map the eigenvalues back through the corresponding Mobius transformation. That recovers the cheap
symmetric route for a class of pencils the simple test rejects.

### 5.2 Quadratic eigenvalue problems.

Damping gives `(lambda^2 M + lambda C + K) x = 0`, which is not a pencil because of the
`lambda^2`. Setting `y = lambda x` linearises it into a pencil of twice the size. Two standard
choices:

```
first companion:   [ C  K ]  -  lambda [ -M  0 ]
                   [ -I 0 ]            [  0  I ]

second companion:  [ C  I ]  -  lambda [ -M  0 ]
                   [ K  0 ]            [  0  I ]
```

Both have the same `2n` eigenvalues, which are the roots of `det(lambda^2 M + lambda C + K)`, and
they differ in conditioning.

```python
def linearise(M, C, K, kind="first"):
    n = M.shape[0]
    I, Z = np.eye(n), np.zeros((n, n))
    if kind == "first":
        return np.block([[C, K], [-I, Z]]), np.block([[-M, Z], [Z, I]])
    return np.block([[C, I], [K, Z]]), np.block([[-M, Z], [Z, I]])
```

Measured on `M = I`, a random symmetric `C` of norm about 0.1, and `K` scaled over nine orders. The
residual is the backward error of the recovered eigenvalue in the original quadratic:

| scaling of `K` | first companion residual | second companion residual | ratio |
|---|---|---|---|
| 1e0 | 1.91e1 | 2.19e0 | 8.7x |
| 1e3 | 2.20e4 | 9.91e1 | 222x |
| 1e6 | 1.48e7 | 1.49e3 | 9933x |
| 1e9 | 2.54e10 | 1.79e5 | 141900x |

**The second companion form is better, and the gap grows with the scaling of `K`**, reaching five
orders of magnitude at `K` of size 1e9.

The reason is where `K` sits. In the first form `K` appears in the top right block alongside `C`,
so the linearised matrix has entries of size `||K||` and size `||C||` in the same rows, and the
small ones are lost. In the second form `K` sits in the bottom left with the identity above it, so
the row scalings are less extreme.

The general lesson, due to Higham, Mackey and Tisseur, is that **the choice of linearisation is
part of the numerical method**, not a notational convenience, and the right choice depends on the
relative sizes of `M`, `C` and `K`. Their practical recommendation is to **scale first**: replace
`lambda` by `gamma mu` and multiply through by a constant, choosing `gamma` so the three
coefficient matrices have comparable norms, then linearise. With good scaling the two companion
forms become comparable and both are acceptable. Without it, as the table shows, the choice is
worth five orders of magnitude.

There is also a structural cost worth naming: for `M`, `C`, `K` all symmetric, the quadratic
problem has eigenvalues in conjugate pairs and a symmetry that neither companion form preserves.
Symmetric linearisations exist and keep it, at the price of a possibly indefinite `B`, which then
needs QZ or the Crawford treatment of 40.5.1.

### 5.3 Structure preserving methods.

A symmetric definite pencil `(A, B)` has real eigenvalues and `B`-orthogonal eigenvectors, by
40.2.2. QZ throws both away: it treats the pair as general, produces a generalized Schur form with
no symmetry, returns eigenvalues that are complex to within rounding rather than exactly real, and
gives eigenvectors with no `B`-orthogonality.

The Cholesky reduction preserves the structure but is limited by `kappa(B)`, since it forms
`L^{-1} A L^{-T}` and the inverses of an ill conditioned `L` amplify.

**One method that does better: the symmetric indefinite generalized Jacobi method**, or more
practically, the **Falk-Langemeyer** and **Veselic** algorithms. These apply non-orthogonal but
`J`-orthogonal transformations `x -> S x` chosen to annihilate the `(p, q)` entry of **both** `A`
and `B` at once. Each step is a 2 by 2 congruence applied to both matrices simultaneously,
symmetry is preserved exactly at every step, and the eigenvalues stay real by construction.

What it buys over the Cholesky reduction when `B` is badly conditioned: the accuracy bound is in
terms of the **scaled** condition numbers of `A` and `B`, the same `kappa(B)` in the Demmel-Veselic
sense of 38.2.5 rather than the ordinary condition number. For a pencil where both matrices are
badly scaled but well conditioned after scaling, which is the common case for a finite element
model with mixed units, that is the difference between a few correct digits and full accuracy. It
inherits the Demmel-Veselic guarantee to the pencil setting, and it does so because it never
forms an inverse, never subtracts, and keeps both matrices symmetric throughout.

The cost is that of Jacobi, so several sweeps of `O(n^3)`, meaning it is slower than the Cholesky
route on well scaled problems and slower than QZ in raw flops. It is used where the accuracy is
worth it, and its parallelism, as in 38.5.1, makes the flop count less decisive than it looks.

---
## Lesson 41, SVD Theory

### Level 1, conceptual

### 1.1 The SVD always exists and an eigendecomposition need not. Give the two conditions the second one needs, and a matrix failing each.

An eigendecomposition `A = V D V^{-1}` needs, first, that `A` is **square**, and second, that `A`
is **diagonalizable**, meaning it has `n` linearly independent eigenvectors.

A matrix failing the first: any rectangular matrix, for instance `[[1, 2, 3]]`. Eigenvalues are not
defined for it at all, because `A x = lambda x` requires `A x` and `x` to live in the same space.
Its SVD exists and has one singular value, `sqrt(14)`.

A matrix failing the second: the Jordan block `[[1, 1], [0, 1]]`. Its only eigenvalue is 1 with
algebraic multiplicity 2 and geometric multiplicity 1, so there is no basis of eigenvectors and no
`V` to invert. Its SVD exists, with singular values `(1 + sqrt(5))/2 = 1.618` and
`(sqrt(5) - 1)/2 = 0.618`.

The SVD needs neither condition because it uses **two** orthonormal bases, one in each space,
rather than one basis serving both roles. That extra freedom is exactly what removes both
obstructions.

### 1.2 Sampling the unit sphere verified the geometric claim at `n = 2` and failed at `n = 20`. What went wrong?

Nothing went wrong with the claim. What failed was the sampling.

The claim is that `max ||A x||` over the unit sphere equals `sigma_1`, attained at `x = v_1`. To
verify it by sampling you need a sample point close to `v_1`. In `n` dimensions a random unit
vector has expected squared overlap `1/n` with any fixed direction, so its overlap with `v_1` is
about `1/sqrt(n)`. The measured shortfall is in exercise 41.4.1: at `n = 2` the best of 10000
samples reaches 1.0000 of `sigma_1`, and at `n = 100` it reaches only 0.6835.

This is concentration of measure. In high dimensions almost all of the sphere's area sits near any
equator, so almost every random direction is nearly orthogonal to `v_1`. Sampling explores the
typical part of the sphere and the maximum lives in an atypical part.

It says nothing against the claim, which is a theorem. It says that **sampling is the wrong
verification method for an extremal property in high dimensions**, and the right one is to check a
defining identity, here `A v_1 = sigma_1 u_1` and `||A||_2 = sigma_1`, which holds exactly at any
size.

### 1.3 Two libraries return different right singular vectors for the same matrix. Give two circumstances in which both are correct.

**One, a repeated singular value.** If `sigma_i = sigma_{i+1}`, any orthonormal basis of the
corresponding two dimensional subspace is a valid pair of singular vectors. There is no canonical
choice, and different implementations return different ones depending on rounding. This is the
same situation as 38.5.3's clustered eigenvalues, and it is not a defect.

**Two, the sign.** Even for a simple singular value, `(u_i, v_i)` and `(-u_i, -v_i)` are both
valid, since `A v_i = sigma_i u_i` becomes `A(-v_i) = sigma_i (-u_i)`. Both libraries are right
and they made opposite sign choices. LAPACK does not normalise the sign, so this happens routinely.

A third circumstance worth adding: for a **wide** matrix with `m < n`, the last `n - m` right
singular vectors span the null space, and any orthonormal basis of it is valid. That is a repeated
singular value at zero, so it is really the first case again.

The practical consequence for testing is that a test comparing singular vectors entry by entry is
wrong. The right tests are the defining identities, `||A - U S V^T||`, `||U^T U - I||`,
`||V^T V - I||`, and the singular values themselves, all of which are unique.

### Level 2, mathematical

### 2.1 Prove that every matrix has an SVD.

By induction on `min(m, n)`. Assume `m >= n`, otherwise transpose.

The function `x -> ||A x||` is continuous on the unit sphere, which is compact, so it attains a
maximum. Let `sigma_1 = max ||A x||` over `||x|| = 1`, attained at a unit `v_1`, and set
`u_1 = A v_1 / sigma_1` if `sigma_1 > 0`, or any unit vector if `sigma_1 = 0`. Then `u_1` is a
unit vector and `A v_1 = sigma_1 u_1`.

Extend `v_1` to an orthonormal basis giving `V_1 = [v_1, V_1']`, and `u_1` to `U_1 = [u_1, U_1']`.
Then

```
U_1^T A V_1 = [ sigma_1   w^T ]
              [    0      A_1 ]
```

The zero block is because `U_1'^T A v_1 = sigma_1 U_1'^T u_1 = 0`.

**Claim: `w = 0`.** Consider the unit vector `x` proportional to `(sigma_1, w^T)^T` in the new
coordinates. Then

```
|| U_1^T A V_1 x ||  >=  first component  =  (sigma_1^2 + w^T w) / sqrt(sigma_1^2 + w^T w)
                                          =  sqrt(sigma_1^2 + w^T w)
```

But `||U_1^T A V_1 x|| = ||A V_1 x||` and `V_1 x` is a unit vector, so this is at most `sigma_1` by
the definition of `sigma_1`. Therefore `sqrt(sigma_1^2 + w^T w) <= sigma_1`, forcing `w = 0`.

So `U_1^T A V_1 = diag(sigma_1, A_1)` block diagonally, and induction on `A_1`, which is
`(m-1)` by `(n-1)`, completes the proof. The singular values come out in decreasing order because
each step takes the maximum of what remains.

### 2.2 Prove the singular values are unique, and the vectors are unique up to sign exactly when the values are simple.

**Uniqueness of the values.** If `A = U S V^T` then `A^T A = V S^T S V^T`, so the `sigma_i^2` are
the eigenvalues of `A^T A`, which are determined by `A` alone. Since the `sigma_i` are
non-negative, they are determined too. Alternatively use the variational characterisation

```
sigma_k = max over k-dimensional S of min over unit x in S of ||A x||
```

which mentions no decomposition at all.

**Uniqueness of the vectors.** Suppose `sigma_i` is simple, meaning distinct from every other
singular value. Then `sigma_i^2` is a simple eigenvalue of `A^T A`, so its eigenspace is one
dimensional and `v_i` is determined up to a scalar; being unit, up to a sign. Then
`u_i = A v_i / sigma_i` is determined by `v_i`, so flipping `v_i` flips `u_i` and the pair is
determined up to a common sign.

Conversely if `sigma_i = sigma_j` for `i != j`, the eigenspace of `A^T A` for that value is at
least two dimensional, and any orthonormal basis of it gives valid right singular vectors, so the
freedom is a full orthogonal group of dimension at least 1 rather than just a sign.

For a zero singular value, `u_i` is not determined by `v_i` at all, since the formula divides by
`sigma_i`. Any unit vector orthogonal to the previous `u`'s will do, which is the null space
freedom of 41.1.3.

### 2.3 Prove all four subspace identities, and hence rank-nullity.

Let `A = U S V^T` with rank `r`, so `sigma_1 >= ... >= sigma_r > 0` and the rest are zero.

**range(A) = span of `u_1, ..., u_r`.** For any `x`, `A x = U S V^T x = sum_i sigma_i (v_i^T x) u_i`,
which lies in that span since `sigma_i = 0` for `i > r`. Conversely `A v_i = sigma_i u_i` gives
each `u_i` with `i <= r` as a multiple of something in the range.

**null(A) = span of `v_{r+1}, ..., v_n`.** If `A x = 0` then `sum_i sigma_i (v_i^T x) u_i = 0`, and
the `u_i` are independent, so `sigma_i (v_i^T x) = 0` for each `i`, so `v_i^T x = 0` for `i <= r`.
That places `x` in the span of the rest. Conversely `A v_i = 0` for `i > r`.

**range(A^T) = span of `v_1, ..., v_r`** and **null(A^T) = span of `u_{r+1}, ..., u_m`** follow by
applying the first two to `A^T = V S^T U^T`.

**Rank-nullity.** `dim range(A) = r` and `dim null(A) = n - r`, so they sum to `n`. The theorem
falls out of the counting, and the SVD makes it visible: the `n` right singular vectors split into
`r` that map to something and `n - r` that map to zero.

The SVD also gives the orthogonality for free: `range(A^T)` and `null(A)` are orthogonal
complements in `R^n`, since they are spanned by complementary subsets of the same orthonormal
basis. That is the fundamental theorem of linear algebra, and proving it any other way takes real
work.

### 2.4 Prove `||A||_2 = sigma_1`, `kappa_2 = sigma_1/sigma_n`, and `||A||_F = sqrt(sum sigma_i^2)`.

For the 2-norm, `||A||_2 = max ||A x|| / ||x||`. Since `U` and `V` are orthogonal they preserve
norms, so with `y = V^T x`,

```
||A x|| = ||U S V^T x|| = ||S y||,   ||x|| = ||y||
```

and `||S y||^2 = sum sigma_i^2 y_i^2 <= sigma_1^2 ||y||^2`, with equality at `y = e_1`. So
`||A||_2 = sigma_1`.

For the condition number, `A^{-1} = V S^{-1} U^T` when `A` is square and nonsingular, and
`S^{-1}` has diagonal `1/sigma_i` with largest entry `1/sigma_n`. So `||A^{-1}||_2 = 1/sigma_n` by
the same argument, and

```
kappa_2(A) = ||A||_2 ||A^{-1}||_2 = sigma_1 / sigma_n
```

For the Frobenius norm, `||A||_F^2 = trace(A^T A) = trace(V S^T S V^T) = trace(S^T S)` using the
cyclic property of the trace and `V^T V = I`. And `trace(S^T S) = sum sigma_i^2`.

The three together are why the SVD is the right tool for anything involving norms or conditioning:
all of them are read directly off the diagonal.

### 2.5 Prove Weyl's inequality for singular values.

**Claim.** `|sigma_k(A + E) - sigma_k(A)| <= ||E||_2` for every `k`.

Use the Courant-Fischer minimax characterisation,

```
sigma_k(A) = max over k-dimensional S of min over unit x in S of ||A x||
```

For any subspace `S` and unit `x` in it, the triangle inequality gives

```
||(A + E) x||  <=  ||A x|| + ||E x||  <=  ||A x|| + ||E||_2
```

Taking the min over `x` in `S` and then the max over `S`,

```
sigma_k(A + E)  <=  sigma_k(A) + ||E||_2
```

Swapping the roles of `A` and `A + E`, with perturbation `-E`, gives the other direction, and
together they give the claim.

**Why the eigenvalue version needs symmetry.** The same argument applied to eigenvalues needs
Courant-Fischer for eigenvalues, which requires the Rayleigh quotient `x^T A x` to characterise
them, and that holds only for symmetric matrices. For a non-symmetric matrix the eigenvalues have
no minimax characterisation and Weyl fails badly: a Jordan block of size `m` perturbed by `eps`
moves its eigenvalues by `eps^{1/m}`, which for `m = 5` and `eps = 1e-10` is `1e-2`, a hundred
million times the perturbation. Exercise 35.5.3 measures exactly this.

**So singular values are always perfectly conditioned and eigenvalues are perfectly conditioned
only when the matrix is normal.** That is the single most important practical difference between
the two decompositions.

### Level 3, computational

### 3.1 The SVD from the eigenvectors of the Jordan-Wielandt matrix.

The Jordan-Wielandt matrix is

```
J = [ 0    A  ]
    [ A^T  0  ]
```

which is symmetric of size `m + n`. Its eigenvalues are `+-sigma_i` together with `|m - n|` zeros,
and its eigenvectors are `(u_i, v_i)/sqrt(2)` and `(u_i, -v_i)/sqrt(2)`.

```python
def svd_from_jordan_wielandt(A, tol=1e-12):
    """Recover U, s and V from the eigenvectors of the symmetric matrix J, not just the values.
    The eigenvector for +sigma_i is (u_i, v_i)/sqrt(2), so the two halves of it, rescaled by
    sqrt(2), ARE the singular vectors. Everything is sized from A."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    k = min(m, n)
    J = np.block([[np.zeros((m, m)), A], [A.T, np.zeros((n, n))]])
    w, Q = np.linalg.eigh(J)
    order = np.argsort(w)[::-1][:k]              # the k largest, which are +sigma_1..+sigma_k
    s = w[order]
    U = np.sqrt(2.0) * Q[:m, order]
    V = np.sqrt(2.0) * Q[m:, order]
    for i in range(k):                            # fix the sign so that A v_i = sigma_i u_i
        if s[i] > tol and float(U[:, i] @ (A @ V[:, i])) < 0:
            V[:, i] = -V[:, i]
    return U, np.abs(s), V.T
```

Verified against all four SVD properties:

| shape | `\|\|A - U S V^T\|\|` | `\|\|U^T U - I\|\|` | `\|\|V^T V - I\|\|` | values against LAPACK |
|---|---|---|---|---|
| 8 by 8 | 1.55e-14 | 4.70e-14 | 4.66e-14 | 2.66e-15 |
| 20 by 8 | 1.79e-14 | 3.64e-15 | 3.57e-15 | 5.77e-15 |
| 8 by 20 | 2.23e-14 | 3.04e-15 | 3.10e-15 | 3.55e-15 |
| 60 by 30 | 8.39e-14 | 9.55e-15 | 8.48e-15 | 1.24e-14 |

The one subtlety, and it is where an implementation goes wrong: taking the `k` eigenvalues largest
**in modulus** returns the `+-` pair of the top `k/2`, not the top `k` positive ones. The correct
selection is `np.sort(w)[::-1][:k]`, the `k` algebraically largest. Measured on an 8 by 8 matrix:

```
correct, algebraically largest : max error 4.44e-15
wrong, largest in modulus      : max error 1.89e+00
```

The wrong selection is not slightly worse. It returns a completely different set of numbers.

The reason this route is accurate where the `A^T A` route is not: `J` is formed from the entries
of `A` **without any multiplication**, so no information is squared away. Exercise 41.4.2 measures
the difference to be a factor of `kappa` in the error.

### 3.2 The polar decomposition and orthogonal Procrustes.

`A = Q P` with `Q` having orthonormal columns and `P` symmetric positive semidefinite. From the
SVD, `Q = U V^T` and `P = V S V^T`.

```python
def polar(A):
    """Q is the NEAREST matrix with orthonormal columns to A, in both the 2-norm and the
    Frobenius norm. P = sqrt(A^T A) is the symmetric positive semidefinite stretch."""
    out = sv.svd(np.asarray(A, dtype=float))
    return out.U @ out.Vt, out.Vt.T @ np.diag(out.s) @ out.Vt


def procrustes(A, B):
    """argmin over orthogonal Q of ||A Q - B||_F is the orthogonal factor of the polar
    decomposition of A^T B."""
    return polar(np.asarray(A, dtype=float).T @ np.asarray(B, dtype=float))[0]
```

Measured:

| shape | `\|\|A - QP\|\|` | `Q` orthonormal | `P` symmetric | smallest eigenvalue of `P` | nearest of 400 random |
|---|---|---|---|---|---|
| 6 by 6 | 4.58e-15 | 2.49e-15 | 3.91e-16 | 0.0602 | yes |
| 10 by 4 | 3.21e-15 | 1.81e-15 | 4.98e-16 | 1.1114 | yes |
| 30 by 30 | 6.67e-14 | 9.82e-15 | 3.73e-15 | 0.1486 | yes |
| 50 by 12 | 3.92e-14 | 4.43e-15 | 2.76e-15 | 3.6111 | yes |

`P` comes out positive definite in all four cases, and `Q` beats the best of 400 random matrices
with orthonormal columns every time, which is the nearness property.

For Procrustes, with `B = A Q_true + noise` at noise level 1e-3:

```
recovered Q to 1.41e-3, residual 0.0118, best of 2000 random orthogonal Q gives 12.3382
```

The recovered `Q` is accurate to the noise level, and the residual is a thousand times smaller than
anything random search finds.

**Why the polar `Q` solves Procrustes.** Minimising `||A Q - B||_F^2 = ||A||_F^2 + ||B||_F^2 -
2 trace(Q^T A^T B)` means **maximising** `trace(Q^T M)` with `M = A^T B`. Writing `M = U S V^T`,

```
trace(Q^T U S V^T) = trace(V^T Q^T U S) = trace(Z S) = sum_i sigma_i z_ii
```

where `Z = V^T Q^T U` is orthogonal, so `|z_ii| <= 1` and the sum is at most `sum sigma_i`, attained
when `Z = I`, that is `Q = U V^T`. That is the polar factor.

This is the algorithm behind aligning two point clouds, comparing protein structures, and finding
the rotation between two coordinate frames.

### 3.3 The CS decomposition and principal angles.

If `Q = [Q_1; Q_2]` has orthonormal columns, then `Q_1 = U_1 C Z^T` and `Q_2 = U_2 S Z^T` with the
**same** `Z` and `C^2 + S^2 = I`. The cosines are the cosines of the principal angles between the
column space of `Q` and the coordinate subspace of the first block.

```python
def cs_decomposition(Q1, Q2):
    """One SVD of the first block fixes Z, and the second block is then automatically diagonalised
    by the same Z, with the sines as its column norms. That C^2 + S^2 = I is forced by the
    orthonormality of the stacked matrix, not imposed."""
    U1, c, Zt = np.linalg.svd(np.asarray(Q1, dtype=float), full_matrices=False)
    B = np.asarray(Q2, dtype=float) @ Zt.T
    s = np.linalg.norm(B, axis=0)
    U2 = B / np.where(s > 0, s, 1.0)
    return U1, np.clip(c, 0.0, 1.0), s, U2, Zt
```

Measured against the principal angles computed directly by lesson 16's method:

| split | `\|\|C^2 + S^2 - I\|\|` | `\|\|Q_1 - U_1 C Z^T\|\|` | `\|\|Q_2 - U_2 S Z^T\|\|` | angles agree to |
|---|---|---|---|---|
| 5 + 4, `k = 3` | 6.66e-16 | 3.71e-16 | 4.05e-16 | 1.11e-15 |
| 8 + 6, `k = 4` | 8.88e-16 | 1.11e-15 | 7.16e-16 | 1.39e-15 |
| 12 + 9, `k = 5` | 8.88e-16 | 1.95e-15 | 6.28e-16 | 1.22e-15 |
| 20 + 7, `k = 6` | 1.11e-15 | 1.56e-15 | 9.04e-16 | 7.49e-16 |

All four identities hold to machine precision at every split, and the CS cosines reproduce lesson
16's principal angles exactly.

The relationship to lesson 16 is not an analogy but an identity. The principal angles between
`range(Q)` and the coordinate subspace `span{e_1..e_p}` come from the singular values of
`E^T Q = Q_1`, and the CS decomposition **is** the SVD of `Q_1` with the extra information that
the same `Z` works for `Q_2`. The `C^2 + S^2 = I` relation says the two subspaces see complementary
parts of the same angles, which is why an angle near zero in one block is an angle near 90 degrees
in the other.

The numerically important use is that computing small angles from `C` loses accuracy, because
`arccos` is ill conditioned near 1, while computing them from `S` is accurate. So one takes the
angle from whichever of `c` and `s` is smaller, and the CS decomposition provides both.

### Level 4, experimental

### 4.1 Random sampling against the dimension.

Best of 10000 random unit vectors, as a fraction of `sigma_1`:

| `n` | best ratio reached | shortfall |
|---|---|---|
| 2 | 1.0000 | 0.0000 |
| 5 | 0.9967 | 0.0033 |
| 10 | 0.9309 | 0.0691 |
| 20 | 0.8221 | 0.1779 |
| 50 | 0.7652 | 0.2348 |
| 100 | 0.6835 | 0.3165 |

Fitted: shortfall `~ n^{1.33}` over the range `n = 5` to 100.

**The theoretical picture.** A random unit vector `x` in `R^n` has `E[(v_1^T x)^2] = 1/n`, so a
typical sample has overlap about `1/sqrt(n)` with the top singular direction and gives
`||A x||` well below `sigma_1`. Taking the best of `N` samples helps only logarithmically: the
maximum of `N` samples of `(v_1^T x)^2` is about `2 log(N) / n` for large `n`, so the best ratio
behaves like `sqrt(2 log N / n)` in the regime where the spectrum is flat, and better when the
spectrum decays.

The measured decay is faster than `1/sqrt(n)` would give for the ratio itself, because these are
random matrices whose spectra spread out with `n`, but the direction and the mechanism are exactly
concentration of measure.

The practical lesson matches 41.1.2: **an extremal property cannot be verified by sampling in high
dimensions.** The randomised methods of lesson 43 work not by sampling the sphere but by sampling
a whole `k + p` dimensional subspace at once, which is a different and much better conditioned
question.

### 4.2 The accuracy of both eigenvalue routes against `kappa`.

| `kappa` | `A^T A` route | Jordan-Wielandt | ratio |
|---|---|---|---|
| 1e2 | 2.10e-13 | 3.99e-15 | 5.3e1 |
| 1e4 | 1.36e-9 | 5.69e-14 | 2.4e4 |
| 1e6 | 6.57e-6 | 3.38e-12 | 1.9e6 |
| 1e8 | 1.33e-1 | 3.69e-10 | 3.6e8 |
| 1e10 | 1.00e0 | 4.36e-7 | 2.3e6 |
| 1e12 | 1.75e0 | 5.41e-6 | 3.2e5 |
| 1e14 | 2.65e2 | 1.24e-3 | 2.1e5 |

Fitting only the unsaturated points, over `kappa` from 1e2 to 1e7 with 11 points:

```
A^T A route     : relative error ~ kappa^1.93
Jordan-Wielandt : relative error ~ kappa^0.74
```

and over the wider range where the Jordan-Wielandt error is well above its floor, its exponent is
0.99.

**One exponent is twice the other, and the factor of 2 is the squaring.** Forming `A^T A` squares
the singular values, so it squares the condition number, so an error proportional to
`eps * kappa(A^T A) = eps * kappa(A)^2` appears. The Jordan-Wielandt matrix is assembled from the
entries of `A` with no multiplication at all, so its condition number is `kappa(A)` and its error
is `eps * kappa(A)`.

Two things about the fit deserve stating plainly. The `A^T A` exponent is robustly 1.87 to 1.93
across every reasonable fitting window, so the quadratic behaviour is solid. The
Jordan-Wielandt exponent moves between 0.74 and 0.99 depending on the window, because at low
`kappa` its error sits on the machine precision floor and flattens the curve. The honest statement
is that the first is quadratic, the second is linear, and the measured ratio is between 1.9 and
2.6 depending on where you fit.

The `A^T A` route reaches relative error 1, meaning total loss, at `kappa` about 1e10. The
theoretical breakdown point is `kappa = 1/sqrt(eps) = 6.7e7`, and the measured one is within a
couple of orders of that, the difference being that "relative error 1" is a coarse threshold.

### 4.3 Singular vector sensitivity against the gap, and the Wedin bound.

Perturbation of norm 1e-8, measuring how far the top right singular vector rotates:

| gap | measured rotation | Wedin bound `eps/gap` | bound overstates by |
|---|---|---|---|
| 1e-1 | 2.249e-8 | 1.00e-7 | 4.45x |
| 1e-2 | 6.337e-8 | 1.00e-6 | 15.78x |
| 1e-3 | 4.945e-7 | 1.00e-5 | 20.22x |
| 1e-4 | 1.327e-5 | 1.00e-4 | 7.54x |
| 1e-5 | 1.317e-4 | 1.00e-3 | 7.60x |

The bound holds at every gap and the measured rotation grows as the gap shrinks, by four orders of
magnitude across the table while the perturbation stays fixed at 1e-8.

The overstatement of 4x to 20x is the usual worst case gap: Wedin's bound is attained by the
perturbation direction that maximally couples the two singular subspaces, and a random direction
overlaps that one only partially.

The contrast with the singular **values** is the point. By Weyl, 41.2.5, the values move by at most
1e-8 regardless of the gap, with no dependence on it at all. The vectors move by `eps/gap`, which
is unbounded. So a matrix can have perfectly determined singular values and completely undetermined
singular vectors, and that is not an anomaly but the normal situation whenever two singular values
are close.

### Level 5, advanced

### 5.1 Why the SVD is not the eigendecomposition of `A^T A`.

They have the same singular values, and everything else differs.

**What is different in accuracy.** Forming `A^T A` costs `kappa` squared, measured in 41.4.2 as
`kappa^1.93` against `kappa^0.99`. Concretely, at `kappa = 1e8` the `A^T A` route has relative
error 0.13, meaning **no correct digits** in the small singular values, while the direct route has
3.7e-10. The information is destroyed by the multiplication, before any eigensolver runs, and no
amount of care in the eigensolver recovers it. A singular value below `sqrt(eps) sigma_1` is
rounded away entirely when `A^T A` is formed, because `sigma_i^2` underflows relative to
`sigma_1^2`.

**What is different in the vectors.** The eigendecomposition of `A^T A` gives `V` and nothing else.
Recovering `U` from `u_i = A v_i / sigma_i` divides by `sigma_i`, so for small singular values it
divides by a number that is itself inaccurate, and the resulting `u_i` are neither accurate nor
orthogonal. The SVD produces `U` and `V` together, both orthogonal to working precision, with no
division.

**What is different structurally.** `A^T A` is `n` by `n` and discards all information about the
`m` dimensional side. For `m` much larger than `n` it is a large compression, which is sometimes
what you want, but it means the four fundamental subspaces of 41.2.3 are no longer all available:
`range(A)` and `null(A^T)` live in `R^m` and `A^T A` cannot see them.

**The one thing the sentence is good for** is theory. Every proof about singular values can be
routed through `A^T A`, and 41.2.2's uniqueness proof does exactly that. The sentence is true and
useless: true as a statement about exact arithmetic, useless as an algorithm.

### 5.2 Singular vectors are not perfectly conditioned.

**Wedin's theorem.** Let `A` have singular value decomposition with the singular values split into
two groups separated by a gap `delta`, and let `E` be a perturbation. Then the angle `theta`
between the corresponding singular subspaces of `A` and `A + E` satisfies

```
sin theta  <=  max( ||E v||, ||E^T u|| ) / ( delta - ||E|| )
```

so roughly `||E|| / delta`. The gap is between the group of interest and everything else, so for a
single singular vector it is the distance from `sigma_i` to the nearest other singular value.

**A matrix where a tiny perturbation rotates a singular vector 90 degrees.** Take
`A = diag(1 + delta, 1)` and `E = diag(0, 2 delta + 1e-12)`:

| `delta` | vector turns | top singular value moves by | Weyl allows |
|---|---|---|---|
| 1e-6 | 90.0 degrees | 1.00e-6 | 2.00e-6 |
| 1e-10 | 90.0 degrees | 1.01e-10 | 2.01e-10 |
| 1e-14 | 90.0 degrees | 1.01e-12 | 1.02e-12 |
| 0 exactly | 90.0 degrees | 1.00e-12 | 1.00e-12 |

The top right singular vector goes from `e_1` to `e_2`, a full 90 degrees, for a perturbation of
size 1e-12 in the last row.

The mechanism is simple: before the perturbation the larger singular value belongs to the first
coordinate, after it belongs to the second, so the top singular vector swaps. Nothing rotated
gradually; the labels exchanged.

**Reconciling with Weyl.** There is nothing to reconcile, because the two theorems talk about
different objects. Weyl says the singular **values** move by at most `||E||`, and the table confirms
it: the top singular value moved by 1e-12 when `||E|| = 1e-12`. Wedin says the singular
**vectors** move by at most `||E|| / delta`, which for `delta` of 1e-14 is `1e-12/1e-14 = 100`,
and since `sin theta <= 1` always, the bound is vacuous, which is Wedin correctly saying it can
tell you nothing.

The deeper point is that the individual vector is not the right object when the values are close.
The **subspace** spanned by both is perfectly well conditioned: the plane spanned by `e_1` and
`e_2` does not move at all under this perturbation. That is the same resolution as 38.5.3's
clustered eigenvalues, and the same practical rule follows: use the subspace, not the vector.

### 5.3 The SVD of a product.

Given the SVDs of `A` and `B`, what can be said about `AB`? Less than one would hope, and what can
be said is sharp.

**What is false.** `sigma_i(AB)` is **not** `sigma_i(A) sigma_i(B)`. The singular vectors of `A`
and `B` are unrelated, so the product's action is a composition of two different rotations with two
different stretches, and there is no reason for the extremes to align.

**What is true.** The submultiplicative bounds

```
sigma_1(AB)  <=  sigma_1(A) sigma_1(B)
sigma_n(AB)  >=  sigma_n(A) sigma_n(B)
```

and, more sharply, **Horn's log-majorisation**: for every `k`,

```
prod over i <= k of sigma_i(AB)  <=  prod over i <= k of sigma_i(A) sigma_i(B)
```

with equality at `k = n` when both are square, since `det(AB) = det(A) det(B)`.

Measured:

| shape | `sigma_1(AB) / (sigma_1(A) sigma_1(B))` | `sigma_n(AB) / (sigma_n(A) sigma_n(B))` | log-majorisation holds |
|---|---|---|---|
| 4 by 4 | 0.8407 | 1.1410 | yes |
| 8 by 8 | 0.8563 | 2.0985 | yes |
| 16 by 16 | 0.6528 | 14.6471 | yes |

Both bounds hold with room to spare, and the slack grows with `n`, which is the misalignment of the
two sets of singular vectors becoming more pronounced in higher dimensions.

**The connection to lesson 31 exercise 5.3, the QR of a product.** The situation is the same and
the resolution is the same. There, `qr(AB)` cannot be assembled from `qr(A)` and `qr(B)`, because
`Q_A R_A Q_B R_B` has a `R_A Q_B` in the middle that is not triangular. Here, `U_A S_A V_A^T U_B
S_B V_B^T` has a `V_A^T U_B` in the middle that is not diagonal. In both cases the obstruction is
a single orthogonal factor sitting between the two structured parts, and in both cases the honest
answer is that a fresh factorization of the product is required.

**What can be done instead.** The **product SVD**, exercise 42.5.3, computes the singular values of
`AB` without forming the product, by an implicit algorithm working on the two factors
simultaneously. It exists precisely because the naive composition fails and forming `AB` can lose
accuracy when the factors are ill conditioned.

---

## Lesson 42, Computing the SVD

### Level 1, conceptual

### 1.1 "The singular values of `A` are the square roots of the eigenvalues of `A^T A`." Why is that sentence true and useless?

True: `A^T A = V S^2 V^T` is an eigendecomposition with eigenvalues `sigma_i^2`, so the singular
values are the square roots. It is the standard proof of existence and it is correct.

Useless as an algorithm because forming `A^T A` squares the condition number and destroys the small
singular values. Exercise 41.4.2 measures it: the `A^T A` route has relative error growing like
`kappa^1.93` against `kappa^0.99` for a route that does not form the product, and at
`kappa = 1e8` it has no correct digits at all in the small values.

The threshold is `kappa = 1/sqrt(eps) = 6.7e7`. Above it, `sigma_n^2 / sigma_1^2` is below `eps`,
so `sigma_n^2` is lost in the rounding of `A^T A` no matter how it is computed, and taking the
square root of a number that is entirely rounding gives a number that is entirely rounding.

The sentence is a theorem about exact arithmetic being read as a recipe for floating point
arithmetic, which is the recurring mistake of the whole subject.

### 1.2 What does "implicit" mean in "implicit QR sweep", and what is being avoided?

"Implicit" means the shifted matrix `B^T B - mu I` is never formed. The sweep computes only the
**first column** of `B^T B - mu I`, which needs two entries of `B`, uses that to determine the
first Givens rotation, applies it, and then chases the resulting bulge down the bidiagonal with
rotations determined entirely by the entries already present.

Two things are avoided, and they are different.

**`B^T B` is never formed**, which avoids the squaring of 42.1.1. This is the important one for the
SVD specifically. The sweep is mathematically a QR step on `B^T B` but arithmetically it touches
only `B`, so the relative accuracy of the small singular values survives.

**`B^T B - mu I` is never formed**, which avoids the cancellation of 37.5.2. When `mu` is close to
an eigenvalue, subtracting it from the diagonal is a difference of nearly equal numbers and loses
the very information the step is trying to isolate.

The justification that this works is the **implicit Q theorem**: if two orthogonal similarities
reduce a matrix to unreduced Hessenberg form and their first columns agree, then the results agree
up to a diagonal sign matrix. So determining the first rotation from the first column and then
restoring the structure is enough to pin down the whole step.

### 1.3 The zero-shift sweep has a relative accuracy guarantee and measured no better than the shifted one. Is the guarantee worth anything?

Yes, and the measurement in this repository shows exactly where.

On column-graded bidiagonal matrices with a spread of 1e20, both variants reach 3.02e-15 once the
deflation criteria are correct. There is no observed difference, so on those matrices the
guarantee buys nothing.

But on a bidiagonal matrix with **one tiny diagonal entry among neighbours of size 1**, measured in
42.4.2 below, the shifted sweep has relative error **1.49e3** and the zero-shift sweep has
2.01e-13. That is total failure against near perfect accuracy, and it is the guarantee earning its
keep.

So the honest answer has two halves. A guarantee is not an observed difference: on most matrices,
including the ones the exercise suggested, the shifted sweep is just as accurate and 50 to 1400
times faster. And a guarantee is exactly what tells you which matrices are the exception, which no
amount of testing on random matrices would have found. The zero-shift sweep is used inside
LAPACK's `dbdsqr` as a fallback for precisely the cases where the shift would be dangerous, not as
the default.

### Level 2, mathematical

### 2.1 Prove bidiagonalization preserves the singular values, and count the flops.

Golub-Kahan bidiagonalization computes `A = U B V^T` with `U` and `V` orthogonal and `B` upper
bidiagonal, by alternately applying Householder reflectors from the left to clear a column below
the diagonal and from the right to clear a row beyond the superdiagonal.

Preservation is immediate: if `A = U B V^T` with `U`, `V` orthogonal and `B = U_B S V_B^T` is the
SVD of `B`, then

```
A = (U U_B) S (V V_B)^T
```

and `U U_B` and `V V_B` are orthogonal as products of orthogonal matrices. So the singular values of
`A` and `B` are identical, exactly, and the vectors differ by known orthogonal factors.

**Flop count**, for `m >= n`. Step `k` applies a Householder reflector of length `m - k + 1` to the
trailing `n - k + 1` columns, costing `4(m-k)(n-k)`, and one of length `n - k` to the trailing
`m - k` rows, costing `4(m-k)(n-k)`. Summing over `k` from 1 to `n`:

```
without vectors:   4 m n^2 - (4/3) n^3
```

Accumulating `U` costs an extra `4 m^2 n - (4/3) n^3` if the full `U` is wanted, or
`4 m n^2 - (4/3) n^3` for the thin one, and accumulating `V` costs `(4/3) n^3`. So

```
with the thin vectors:   about  8 m n^2 - (4/3) n^3
```

For a square matrix that is `(8/3) n^3` without vectors and about `(20/3) n^3` with them. The
subsequent bidiagonal sweep costs `O(n^2)` without vectors and `O(n^2)` per sweep with them, so
the reduction dominates in both cases, which is why exercise 42.4.3 measures all the bidiagonal
based methods to have the same exponent.

For `m` much larger than `n` there is a better route: do a QR factorization first, at `2mn^2`, and
bidiagonalize the `n` by `n` factor `R` at `(8/3) n^3`. That is the Chan variant and it wins when
`m > 5n/3`.

### 2.2 Show the implicit sweep is one shifted QR step on `B^T B`.

Let `T = B^T B`, which is symmetric tridiagonal. A shifted QR step on `T` computes
`T - mu I = QR` and `T' = RQ + mu I`.

The implicit sweep instead computes a Givens rotation `G_1` whose first column matches that of the
`Q` above, which is the normalised first column of `T - mu I`, namely
`(d_1^2 - mu, d_1 e_1, 0, ..., 0)^T`. It applies `G_1` to the first two **columns of `B`**, which
creates a bulge at position `(2, 1)`, then alternately applies row and column rotations to chase
the bulge off the end, ending with a bidiagonal `B'`.

**The argument.** The net effect on `B` is `B' = P^T B G` for orthogonal `P` and `G`, so

```
B'^T B' = G^T B^T P P^T B G = G^T T G
```

which is an orthogonal similarity of `T`, and `B'^T B'` is tridiagonal because `B'` is bidiagonal.
The first column of `G` is the first column of `G_1`, which was constructed to be the first column
of `Q`.

By the **implicit Q theorem**, two orthogonal matrices that reduce `T` to unreduced tridiagonal form
and have the same first column produce the same result up to a diagonal sign matrix. Both `G` and
`Q` do so. Therefore `G^T T G = D (Q^T T Q) D` for a sign matrix `D`, and since
`Q^T T Q = Q^T (QR + mu I) Q = RQ + mu I = T'`,

```
B'^T B'  =  D T' D
```

which is `T'` up to signs. So the implicit sweep on `B` performs exactly one shifted QR step on
`B^T B`, without ever forming it.

### 2.3 Derive the Wilkinson shift for `B^T B` from the entries of `B`.

`T = B^T B` for upper bidiagonal `B` with diagonal `d` and superdiagonal `e` has

```
T_{kk}      = d_k^2 + e_{k-1}^2       (with e_0 = 0)
T_{k,k+1}   = d_k e_k
```

The Wilkinson shift is the eigenvalue of the trailing 2 by 2 block

```
[ d_{n-1}^2 + e_{n-2}^2    d_{n-1} e_{n-1} ]
[ d_{n-1} e_{n-1}          d_n^2 + e_{n-1}^2 ]
```

closer to the bottom entry. Writing `a = d_{n-1}^2 + e_{n-2}^2`, `b = d_{n-1} e_{n-1}`,
`c = d_n^2 + e_{n-1}^2`, and `delta = (a - c)/2`, the standard stable form is

```
mu = c - sign(delta) * b^2 / ( |delta| + sqrt(delta^2 + b^2) )
```

**Every intermediate quantity is a product of two entries of `B`.** `a`, `b`, `c` are each sums of
products of two entries; `delta` is a difference of two such sums; `b^2` is a product of four
entries. Nothing requires forming any part of `T` beyond these four numbers, and the whole shift
costs about ten flops.

The form above is written to avoid cancellation: the naive quadratic formula
`mu = c + delta - sign(delta) sqrt(delta^2 + b^2)` subtracts two nearly equal numbers when
`|b|` is small compared with `|delta|`, which is exactly the converged case, and the rationalised
version does not.

### 2.4 Prove one-sided Jacobi on the columns of `A` is two-sided Jacobi on `A^T A`.

Two-sided Jacobi on `M = A^T A` applies `M -> J^T M J` where `J` is a rotation in the `(p, q)`
plane chosen so that `(J^T M J)_{pq} = 0`.

One-sided Jacobi applies `A -> A J` with the same `J`. Then

```
(A J)^T (A J) = J^T A^T A J = J^T M J
```

so the Gram matrix of the rotated columns is exactly the two-sided Jacobi update of `M`. The
condition `(J^T M J)_{pq} = 0` says the new columns `p` and `q` are **orthogonal**, and the
rotation angle is determined by

```
tan(2 theta) = 2 M_{pq} / (M_{pp} - M_{qq}) = 2 (a_p . a_q) / (a_p . a_p - a_q . a_q)
```

which needs only three inner products of the columns of `A`, so `M` is never formed.

Since the Gram matrices track exactly, the off diagonal norm of `M` decreases by `2 M_{pq}^2` per
rotation by 38.2.1, and the quadratic convergence of 38.2.2 applies unchanged. When the process
finishes, `A^T A` is diagonal, meaning the columns of the final `A` are orthogonal, so
`A_final = U S` with `U` orthonormal and `S` the column norms, and `V` is the product of all the
rotations.

**What is gained by working on `A` instead of `M`.** The rotation angle is computed from inner
products of columns of `A`, which are accurate to `eps` **relative to the column norms**. Working
on `M` computes them from entries of `M` that were themselves formed by squaring, so their small
entries have already lost accuracy. That is the whole content of the Demmel-Veselic guarantee.

### 2.5 State the Demmel-Kahan theorem, and explain why the shifted version does not satisfy it.

**Theorem (Demmel and Kahan, 1990).** For an upper bidiagonal `B`, the zero-shift QR sweep computes
every singular value with relative error at most `O(n) * eps`, independently of the condition
number of `B` and independently of how many orders of magnitude the singular values span.

**The hypothesis on `B`** is that it is **bidiagonal**, which is not a triviality. A bidiagonal
matrix is determined by `2n - 1` numbers, and each singular value is determined by those numbers to
high **relative** accuracy: small relative changes to `d_i` and `e_i` cause only small relative
changes to every `sigma_j`. That property fails for a general matrix, where a small relative change
to an entry can change a small singular value completely. So the theorem is about a class of
matrices whose parameterisation is well behaved, and the bidiagonalization that produces `B` does
**not** preserve this: the reduction is backward stable in the absolute sense only.

**Why the shifted version does not satisfy it.** With no shift, every rotation is determined by
ratios of entries of `B` and every update is a sum of products of entries with rotation
coefficients, all of the same sign structure. No subtraction of nearly equal quantities occurs
anywhere, so every computed entry has full relative accuracy.

With a shift, the first rotation is determined from `d_1^2 - mu`, which is a subtraction. When `mu`
is close to `d_1^2`, that subtraction cancels, and the direction of the first rotation is computed
with few or no correct digits. The error then propagates through the whole bulge chase.

The measurement in 42.4.2 finds exactly this, and finds it on a specific structure: a matrix with
one tiny diagonal entry among neighbours of size 1, where the shift is `O(1)` and the target
singular value is `O(1e-18)`.

### Level 3, computational

### 3.1 Golub-Reinsch with the vectors accumulated.

The full algorithm: bidiagonalize, accumulating `U` and `V`, then sweep, applying every Givens
rotation to `U` and `V` as it goes.

`nalib.svdcompute.golub_kahan_svd` returns the values only, so the sweep has to be written out
with the two accumulation lines added. Composing the library sweep with the bidiagonalization's
vectors does **not** work: it returns the bidiagonalization's `U` and `V`, which are not the
singular vectors, and the reconstruction identity then fails.

```python
def golub_reinsch(A, tol=1e-14, max_iter=100_000):
    """Each ROW rotation of the bulge chase multiplies U on the right, each COLUMN rotation
    multiplies V on the right, so both factors stay orthogonal by construction and never need
    re-orthogonalizing. Sizes come from A throughout."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    transposed = A.shape[0] < A.shape[1]
    work = A.T if transposed else A
    d, e, U, Vt = sc.bidiagonalize(work, compute_uv=True)
    V = Vt.T
    d, e = d.copy(), e.copy()
    n = d.size
    high = n - 1
    steps = 0
    while high > 0 and steps < int(max_iter):
        if abs(e[high - 1]) <= tol * (abs(d[high]) + abs(d[high - 1])):
            e[high - 1] = 0.0
            high -= 1
            continue
        low = high
        while low > 0 and abs(e[low - 1]) > tol * (abs(d[low]) + abs(d[low - 1])):
            low -= 1
        # a negligible DIAGONAL entry, tested against its own neighbours and not against the
        # largest entry in the block, which is the fix exercise 42.4.2 produced
        zero_at = -1
        for k in range(low, high + 1):
            local = (abs(e[k]) if k < high else 0.0) + (abs(e[k - 1]) if k > low else 0.0)
            if abs(d[k]) <= tol * local:
                zero_at = k
                break
        if 0 <= zero_at < high:
            extra = e[zero_at]
            e[zero_at] = 0.0
            for k in range(zero_at + 1, high + 1):
                c, s = qr.givens_rotation(d[k], extra)
                d[k] = c * d[k] + s * extra
                U[:, [k, zero_at]] = np.column_stack(
                    [c * U[:, k] + s * U[:, zero_at], -s * U[:, k] + c * U[:, zero_at]])
                if k < high:
                    extra = -s * e[k]
                    e[k] = c * e[k]
            steps += 1
            continue
        mu = sc.wilkinson_shift_squared(d[low:high + 1], e[low:high])
        y, z = d[low] * d[low] - mu, d[low] * e[low]
        for k in range(low, high):
            c, s = qr.givens_rotation(y, z)
            if k > low:
                e[k - 1] = c * y + s * z
            t1, t2 = d[k], e[k]
            d[k] = c * t1 + s * t2
            e[k] = -s * t1 + c * t2
            t3 = d[k + 1]
            z = s * t3
            d[k + 1] = c * t3
            V[:, [k, k + 1]] = np.column_stack(
                [c * V[:, k] + s * V[:, k + 1], -s * V[:, k] + c * V[:, k + 1]])
            c, s = qr.givens_rotation(d[k], z)
            d[k] = c * d[k] + s * z
            t1, t2 = e[k], d[k + 1]
            e[k] = c * t1 + s * t2
            d[k + 1] = -s * t1 + c * t2
            U[:, [k, k + 1]] = np.column_stack(
                [c * U[:, k] + s * U[:, k + 1], -s * U[:, k] + c * U[:, k + 1]])
            if k < high - 1:
                t3 = e[k + 1]
                y, z = e[k], s * t3
                e[k + 1] = c * t3
                y = e[k]
        steps += 1
    flip = d < 0                        # a negative value means the sign belongs on u
    d[flip] = -d[flip]
    U[:, flip] = -U[:, flip]
    order = np.argsort(d)[::-1]
    d, U, V = d[order], U[:, order], V[:, order]
    if transposed:
        return V, d, U.T, steps
    return U, d, V.T, steps
```

Verified on all four SVD properties:

| shape | `\|\|A - U S V^T\|\|` | `\|\|U^T U - I\|\|` | `\|\|V^T V - I\|\|` | values against LAPACK | sweeps |
|---|---|---|---|---|---|
| 8 by 8 | 1.43e-14 | 3.50e-15 | 2.29e-15 | 4.44e-15 | 15 |
| 30 by 12 | 6.05e-14 | 4.49e-15 | 4.39e-15 | 7.99e-15 | 22 |
| 12 by 30 | 6.21e-14 | 4.35e-15 | 3.57e-15 | 7.99e-15 | 23 |
| 100 by 40 | 3.03e-13 | 1.07e-14 | 1.19e-14 | 3.91e-14 | 73 |
| 100 by 100 | 7.95e-13 | 3.25e-14 | 3.04e-14 | 2.03e-13 | 165 |

All four hold at every size, and the sweep count is 1.6 to 1.9 per singular value, matching
42.4.1. The two sign lines at the end matter: the sweep can leave a `d_i` negative, and the
convention that singular values are non-negative is restored by moving the sign onto the
corresponding column of `U`, not by taking an absolute value and leaving `U` unchanged.

### 3.2 The dqds algorithm.

Fernando and Parlett's differential quotient difference with shifts. It carries the **squares**
`q_i = d_i^2` and `e_i = off_i^2` and updates them with

```
q'_i = d + e_i,     e'_i = e_i * (q_{i+1} / q'_i),     d <- d * (q_{i+1} / q'_i) - tau
```

started from `d = q_1 - tau`. Every quantity is a product or a sum of positive numbers, so no
cancellation occurs anywhere and the small singular values keep their relative accuracy. This is
what LAPACK uses, as `dlasq`, when only the values are wanted.

```python
def dqds(diag, off, max_iter=200_000):
    q = np.asarray(diag, dtype=float) ** 2
    e = np.concatenate([np.asarray(off, dtype=float) ** 2, [0.0]])
    n = q.size
    out = np.zeros(n)
    high = n
    steps = 0
    sigma = 0.0                      # the shifts applied so far, added back on deflation

    def transform(q, e, high, tau):
        qn, en = np.empty(high), np.empty(high)
        d = q[0] - tau
        for i in range(high - 1):
            qn[i] = d + e[i]
            if qn[i] <= 0.0:
                return None, None
            t = q[i + 1] / qn[i]
            en[i] = e[i] * t
            d = d * t - tau
            if d < 0.0:
                return None, None
        qn[high - 1] = d
        en[high - 1] = 0.0
        return qn, en

    while high > 1 and steps < int(max_iter):
        steps += 1
        if e[high - 2] <= np.finfo(float).eps ** 2 * q[high - 1]:
            out[high - 1] = q[high - 1] + sigma
            high -= 1
            continue
        tau = 0.0 if steps % 8 == 0 else 0.5 * min(q[high - 1],
                                                   max(0.0, q[high - 1] - e[high - 2]))
        qn, en = transform(q, e, high, tau)
        if qn is None:                       # the shift was too large, nothing was committed
            tau = 0.0
            qn, en = transform(q, e, high, 0.0)
        q[:high], e[:high] = qn, en
        e[high - 1] = 0.0
        sigma += tau                         # the block now holds lambda - sigma
    out[:high] = q[:high] + sigma
    return np.sqrt(np.sort(np.clip(out, 0.0, None))[::-1]), steps
```

The shift must keep `d` positive throughout. The step is tried, and if any `d` goes negative
nothing is committed and the step is retried with `tau = 0`, which always succeeds. The accumulated
shift `sigma` is added back when an eigenvalue deflates.

Measured against the Golub-Kahan sweep:

| `n` | dqds steps | dqds relative error | Golub-Kahan relative error | dqds time | GK time | speedup |
|---|---|---|---|---|---|---|
| 20 | 237 | 8.20e-16 | 2.14e-15 | 0.0024s | 0.0039s | 1.6x |
| 40 | 468 | 1.26e-15 | 2.82e-15 | 0.0076s | 0.0160s | 2.1x |
| 80 | 893 | 2.81e-15 | 1.22e-14 | 0.0253s | 0.0482s | 1.9x |
| 160 | 1738 | 2.62e-15 | 1.18e-14 | 0.0925s | 0.1954s | 2.1x |

dqds is **about 2 times faster and about 4 times more accurate**, consistently across sizes. The
speed comes from doing `O(n)` work per step with no rotations and no square roots until the very
end; the accuracy comes from the absence of subtraction.

On a graded bidiagonal, where relative accuracy is the whole point:

| spread | dqds | Golub-Kahan | `A^T A` route |
|---|---|---|---|
| 1e4 | 1.09e-15 | 4.61e-15 | 6.27e-16 |
| 1e8 | 9.41e-16 | 3.76e-15 | 8.36e-16 |
| 1e12 | 4.70e-16 | 3.34e-15 | 3.42e-16 |
| 1e16 | 4.19e-16 | 6.48e-3 | 4.43e-16 |

dqds holds at 4e-16 across the whole range. The Golub-Kahan figure of 6.48e-3 at spread 1e16 comes
from a deflation criterion issue described in 42.4.2, which was found by this measurement and
fixed in `nalib.svdcompute`; after the fix that entry becomes 3.34e-15.

### 3.3 A randomised SVD.

Sample the range with a random matrix, orthonormalise, and take a small SVD. Measured error against
rank `k` and oversampling `p`, on a 400 by 200 matrix with geometrically decaying singular values:

| `k` | `p = 0` | `p = 2` | `p = 5` | `p = 10` | `p = 20` | optimal `sigma_{k+1}` |
|---|---|---|---|---|---|---|
| 5 | 8.906e-1 | 8.469e-1 | 8.436e-1 | 7.785e-1 | 7.095e-1 | 7.067e-1 |
| 10 | 7.778e-1 | 7.352e-1 | 7.424e-1 | 6.360e-1 | 5.114e-1 | 4.995e-1 |
| 20 | 6.090e-1 | 4.821e-1 | 4.542e-1 | 3.694e-1 | 2.519e-1 | 2.495e-1 |
| 40 | 1.562e-1 | 1.564e-1 | 1.990e-1 | 1.263e-1 | 6.294e-2 | 6.223e-2 |

At `p = 20` the error is within 1 percent of the theoretical optimum in every row. At `p = 0` it is
26 percent to 150 percent above it. So a modest oversampling is close to free and buys most of the
gap, which is the Halko-Martinsson-Tropp recommendation of `p = 5` to 10 as a default, and lesson
43 explores the trade against power iterations.

### Level 4, experimental

### 4.1 Sweeps against `n` for both shift strategies.

| `n` | shifted | per singular value | zero-shift | per singular value | ratio |
|---|---|---|---|---|---|
| 10 | 17 | 1.70 | 441 | 44.10 | 25.9x |
| 20 | 35 | 1.75 | 2405 | 120.25 | 68.7x |
| 40 | 68 | 1.70 | 3949 | 98.72 | 58.1x |
| 80 | 133 | 1.66 | 16254 | 203.18 | 122.2x |
| 160 | 261 | 1.63 | 104561 | 653.51 | 400.6x |

Fitted: shifted sweeps `~ n^0.98`, zero-shift `~ n^1.85`.

**The "two per singular value" folklore is confirmed and slightly beaten**: the shifted sweep needs
1.63 to 1.75 sweeps per singular value, essentially constant across a factor of 16 in `n`. Since
each sweep on a bidiagonal costs `O(n)`, the total is `O(n^2)`, which is the standard result.

The zero-shift sweep is a different algorithm asymptotically, not merely a slower one: `n^1.85`
against `n^0.98`, so the ratio grows from 26x at `n = 10` to **401x at `n = 160`** and keeps
growing. That is the cost of the guarantee in 42.1.3, and it is why the zero-shift variant is a
fallback rather than the default.

### 4.2 A matrix where the shifted sweep genuinely loses relative accuracy.

**The premise needed correcting twice before the real answer appeared.**

First attempt, a column-graded bidiagonal. Measured errors at spread 1e16: shifted 6.48e-3,
zero-shift 6.48e-3. **Identical.** The shift was not the cause. Investigating found the cause in
`nalib.svdcompute.golub_kahan_svd`: its test for a negligible diagonal entry compared `|d_k|`
against `tol` times the **largest** diagonal entry in the block. On a graded matrix that discards
legitimately tiny values. At spread 1e16 with `tol = 1e-14`, four of thirty diagonal entries tripped
that test, the smallest being 1e-16, while the true singular values went down to 1e-16.

Changing the test to compare against the neighbouring off diagonals instead:

| spread | shifted, global test | zero, global test | shifted, local test | zero, local test |
|---|---|---|---|---|
| 1e8 | 3.76e-15 | 3.76e-15 | 3.76e-15 | 3.76e-15 |
| 1e12 | 3.34e-15 | 4.36e-15 | 3.34e-15 | 4.36e-15 |
| 1e16 | 6.48e-3 | 6.48e-3 | 3.34e-15 | 3.34e-15 |
| 1e20 | 1.15e-2 | 1.15e-2 | 3.02e-15 | 3.02e-15 |
| 1e24 | 2.26e-2 | 2.26e-2 | 2.95e-15 | 2.58e-15 |

The local test fixes both variants completely and changes nothing on ordinary matrices, where the
errors and the sweep counts come out bit for bit identical. That fix is now in the library, and it
means the deflation criterion, not the shift, was what cost relative accuracy on graded matrices.

**The real adversarial matrix.** With the criterion corrected, a search over structures found it:

| bidiagonal | `sigma_1/sigma_n` | shifted error | zero-shift error | shifted sweeps | zero sweeps |
|---|---|---|---|---|---|
| graded, spread 1e20 | 1.0e20 | 3.02e-15 | 3.02e-15 | 23 | 25 |
| all entries 1 | 3.9e1 | 2.32e-15 | 2.15e-13 | 66 | 3436 |
| `d = 1`, `e = -1` | 3.9e1 | 2.32e-15 | 2.15e-13 | 66 | 3436 |
| `d = 1`, `e = 2` | 2.1e9 | 1.65e-15 | 2.67e-13 | 66 | 4032 |
| random | 1.6e2 | 5.87e-15 | 3.14e-12 | 56 | 77274 |
| **`d = (1,...,1,1e-18)`, `e = 1`** | 1.1e19 | **1.49e3** | **2.01e-13** | 66 | 3329 |

The last row is the answer. A bidiagonal matrix with **one tiny diagonal entry among neighbours of
size 1** breaks the shifted sweep completely, relative error 1.49e3, while the zero-shift sweep
gets 2.01e-13.

**What makes it adversarial.** The smallest singular value is about 1e-18, set by the single tiny
`d_n`. The Wilkinson shift is computed from the trailing 2 by 2 block of `B^T B`, whose entries are
`d_{n-1}^2 + e_{n-2}^2 = 2` and `d_n^2 + e_{n-1}^2 = 1`, both `O(1)`. So the shift is `O(1)` while
the target eigenvalue of `B^T B` is `1e-36`. Forming `d_1^2 - mu` is then a subtraction of two
`O(1)` numbers whose difference must resolve a quantity 36 orders smaller, and every digit is lost.
The zero-shift sweep subtracts nothing and every entry stays a product of ratios, so the tiny value
survives.

Note also that on **every ordinary matrix** in the table the shifted sweep is 100 times **more**
accurate and 50 to 1400 times faster. The zero-shift sweep is not the better algorithm; it is the
one with the guarantee, and the guarantee matters on exactly one row out of six.

### 4.3 Cost of all four methods against `n`.

Without vectors:

| method | fitted exponent | `n = 50` | `n = 100` | `n = 200` | `n = 400` | against LAPACK |
|---|---|---|---|---|---|---|
| `A^T A` eigen | 2.24 | 0.0002s | 0.0014s | 0.0045s | 0.0262s | 1.7x |
| Golub-Kahan | 1.98 | 0.0216s | 0.0798s | 0.3113s | 1.3400s | 87.1x |
| one-sided Jacobi | 2.69 | 0.1324s | 0.6454s | 3.0172s | 39.2399s | 2550.9x |
| LAPACK | 2.23 | 0.0001s | 0.0017s | 0.0035s | 0.0154s | 1.0x |

With vectors:

| method | fitted exponent | `n = 50` | `n = 100` | `n = 200` | `n = 400` | against LAPACK |
|---|---|---|---|---|---|---|
| `A^T A` eigen | 2.21 | 0.0002s | 0.0015s | 0.0045s | 0.0279s | 0.7x |
| Golub-Kahan | 1.97 | 0.0216s | 0.0797s | 0.3099s | 1.3125s | 35.1x |
| one-sided Jacobi | 2.60 | 0.1852s | 0.8993s | 4.2141s | 44.9297s | 1201.3x |
| LAPACK | 1.87 | 0.0007s | 0.0031s | 0.0092s | 0.0374s | 1.0x |

**The crossovers.** There are none in the sense the exercise expects: the ordering is the same at
every size tested. What the table shows instead is the cost of accuracy.

The `A^T A` route is the **fastest**, faster even than LAPACK's SVD when vectors are wanted, at
0.7x. It is also the least accurate, by a factor of `kappa` from 41.4.2. That is the trade in its
starkest form: the cheapest method is the one that throws away half the digits.

One-sided Jacobi is 1200x to 2550x slower here and has the strongest accuracy guarantee. Most of
that factor is Python: the method is `O(n^3)` per sweep done in `O(n^2)` numpy calls, and its
fitted exponent of 2.6 to 2.7 reflects the interpreter. A tuned implementation, LAPACK's `dgesvj`,
runs at 2 to 5 times the bidiagonal route, not 2500.

Adding vectors costs LAPACK a factor of 2.4 and Golub-Kahan essentially nothing in this
implementation, since the sweep count dominates and the bidiagonalization is the same.

### Level 5, advanced

### 5.1 Why the bidiagonal form and not the tridiagonal.

**The relation.** If `A = U B V^T` with `B` upper bidiagonal, then

```
A^T A = V B^T B V^T
```

and `B^T B` is symmetric **tridiagonal**. So bidiagonalizing `A` and tridiagonalizing `A^T A`
produce the same tridiagonal matrix, up to signs: the tridiagonal you would get from
`A^T A` directly is `B^T B`.

More precisely, the Golub-Kahan bidiagonalization of `A` with starting vector `v_1` produces
exactly the tridiagonal that Lanczos on `A^T A` with the same starting vector produces. They are
the same recurrence written on different objects.

**What is lost by going through the tridiagonal.** Everything 42.1.1 says. The entries of `B^T B`
are `d_k^2 + e_{k-1}^2` and `d_k e_k`. Forming them squares the entries, so a `d_k` of size
`1e-10 * d_1` becomes a diagonal entry of size `1e-20 * d_1^2`, which is below `eps` relative to
the largest entry and is rounded to it. The information about that singular value is gone before
any eigensolver starts.

Working on `B` keeps `d_k` itself, at full relative accuracy, and every operation the sweep
performs on it is a rotation whose coefficients are ratios. That is what Demmel-Kahan needs and it
is why the whole subject is built on the bidiagonal form.

There is a second, smaller loss: the tridiagonal route gives `V` and must recover `U` by dividing
by the singular values, which for small ones divides by a number that is itself inaccurate. The
bidiagonal route produces both sets of vectors as accumulated rotations, with no division.

### 5.2 Preconditioned Jacobi.

Drmac and Veselic observed that one-sided Jacobi's slowness comes from starting far from
convergence, and that a cheap preconditioner fixes it without touching the accuracy guarantee.

**The preconditioner.** Compute a QR factorization of `A` with **column pivoting**, `A P = Q R`.
Then run one-sided Jacobi on `R^T`, or on `R` after a second QR factorization. The singular values
of `R` are those of `A` exactly, since `Q` and `P` are orthogonal and a permutation.

**Why it helps.** Column pivoting orders the columns so that `|r_11| >= |r_22| >= ... `, and the
resulting `R` is close to diagonally dominant in a scaled sense, by the rank revealing property of
lesson 33. Its rows are already nearly orthogonal, so Jacobi starts much closer to its fixed point
and needs far fewer sweeps. Drmac and Veselic report a reduction from 8 to 10 sweeps to 2 to 3,
and combined with the `O(n^3)` savings of working on a triangular matrix rather than a full one,
the total comes to within a factor of 2 to 3 of the bidiagonal route.

**Why the guarantee survives.** The QR factorization with column pivoting is backward stable in the
**column-wise** sense: each computed column of `Q R` is the exact column of a matrix whose
corresponding column differs by a small relative amount. That is exactly the hypothesis
Demmel-Veselic needs, since it says the scaled condition number `kappa(B)` is not degraded by the
preconditioner. A preconditioner that mixed columns arbitrarily would destroy this; a QR
factorization does not, because it acts on the left.

The combination is LAPACK's `dgejsv`, which is the routine to use when the small singular values
matter.

### 5.3 The SVD of a product without forming it.

Given `A` and `B`, the singular values of `AB` can be computed without multiplying them. This is
the **product SVD**, or PSVD.

**Why you would want to.** Forming `AB` costs the accuracy of both. If `A` and `B` each have
condition number 1e8, the product has condition number up to 1e16 and its small singular values
are lost in the multiplication, exactly as in 42.1.1. The PSVD keeps them.

**The construction.** There exist orthogonal `U`, `V`, `W` with

```
U^T A W = S_A,     W^T B V = S_B
```

both upper triangular, and the singular values of `AB` are the diagonal entries of `S_A S_B`, that
is, the products `s_A(i) s_B(i)`. The shared `W` is what makes it work: it must simultaneously
triangularize `A` from the right and `B` from the left.

The algorithm is an implicit one: reduce `A` and `B` to triangular form by a coupled sequence of
rotations, then run an implicit sweep that applies each rotation to whichever of the two factors it
belongs to, never forming the product. It is structurally the same idea as the QZ algorithm of
lesson 40, which handles `A - lambda B` by acting on both matrices at once, and as the implicit
bidiagonal sweep, which handles `B^T B` by acting on `B`.

**The relation to lesson 40's generalized problem.** The PSVD is one member of a family. Given
matrices `A` and `B`, the **generalized SVD** computes the singular values of the pair, which are
the eigenvalues of the pencil `(A^T A, B^T B)`, without forming either product. The **product SVD**
computes the singular values of `AB`. The **quotient SVD** computes those of `A B^{-1}`. All three
are handled by the same machinery: find orthogonal transformations that reduce both matrices to
triangular form simultaneously, then read the answer off the diagonals. And all three exist for the
same reason: the naive route forms a product or an inverse and loses accuracy that the data
contains.

---

## Lesson 43, SVD Applications and Low Rank

### Level 1, conceptual

### 1.1 The truncated SVD is optimal in the 2-norm and the Frobenius norm at the same time. Why is that surprising, and what did Mirsky add?

It is surprising because the two norms measure very different things. The 2-norm sees only the
largest singular value of the error, so it cares about the worst direction and ignores everything
else. The Frobenius norm sees the sum of squares of all of them, so it cares about the total. A
matrix minimising one would normally not minimise the other, and for most approximation problems
the optimal answer genuinely does depend on the norm.

The same `A_k` minimises both. In the 2-norm the error is `sigma_{k+1}` and in the Frobenius norm
it is `sqrt(sum_{i>k} sigma_i^2)`, and both are attained by the same truncation.

**What Mirsky added.** He proved it for **every unitarily invariant norm**, meaning every norm with
`||U M V|| = ||M||` for orthogonal `U` and `V`. That class includes the 2-norm, the Frobenius norm,
all the Schatten `p`-norms, the nuclear norm, and the Ky Fan norms. So the result is not a
coincidence of two norms, it is a structural fact: the truncated SVD is optimal in every norm that
does not care about the coordinate system, and the two familiar cases are just the two most used
members of that family.

The proof rests on the fact that a unitarily invariant norm is a symmetric gauge function of the
singular values, so minimising it reduces to a statement about the singular values alone, which
Weyl's inequality settles.

### 1.2 A rank 1 approximation keeps 98.5 percent of the energy and has a 12 percent relative error. Explain, and say which number you would report.

They are the same number, stated in two ways, and the relation is exact:

```
energy captured = sum_{i<=k} sigma_i^2 / sum_i sigma_i^2
relative Frobenius error^2 = sum_{i>k} sigma_i^2 / sum_i sigma_i^2 = 1 - energy
```

So `error = sqrt(1 - 0.985) = sqrt(0.015) = 0.1225`, which is the 12 percent.

The apparent contradiction comes from the **square root**. Energy is measured in squared units and
error in ordinary units, so a small energy deficit becomes a much larger relative error: 1.5
percent of the energy missing is 12 percent of the norm missing.

**Which to report: the error.** Two reasons. It is in the same units as the thing being
approximated, so it is directly comparable to a tolerance and to the noise level. And it is the
conservative statement of the two, so reporting it cannot mislead. "98.5 percent of the energy"
sounds like a 1.5 percent error to almost every reader, and it is not.

Report both if the audience is technical, and always name the norm.

### 1.3 Truncation is used for compression, regularization and denoising. What is different about the three?

The operation is identical. What differs is **what you are trying to remove and how you choose
`k`**.

**Compression.** You want to remove **storage**. The signal is everything, and any truncation is a
loss you accept in exchange for space. `k` is chosen by a budget: how small must the file be, or
what error is tolerable. There is no correct `k`, only a trade off curve, and the relevant
threshold is `k < mn/(m+n)` for the truncation to save anything at all.

**Regularization.** You want to remove **amplification**. The small singular values are real
features of the operator, and the problem is that `1/sigma_i` in the pseudoinverse amplifies noise
in the data by that factor. `k` is chosen by the discrete Picard condition of lesson 32: keep the
components whose data coefficients decay faster than the singular values, discard the rest. The
correct `k` depends on the **noise level in the right hand side**, not on the matrix alone.

**Denoising.** You want to remove **noise that is in the matrix itself**. The matrix is signal plus
noise, and the noise contributes singular values up to the Marchenko-Pastur edge. `k` is chosen at
that edge, and there is a genuinely optimal `k`, computed exactly by Gavish and Donoho in 43.5.2.
Unlike the other two, this one has a right answer.

The sharpest way to see the difference: in compression, discarding a component always costs you
something. In denoising, discarding the right components makes the answer **better**, as measured
against the truth. Exercise 43.4.2 measures the error falling and then rising as `k` grows, and the
minimum is real.

### Level 2, mathematical

### 2.1 Prove Eckart-Young in the 2-norm.

**Claim.** For any `B` of rank at most `k`, `||A - B||_2 >= sigma_{k+1} = ||A - A_k||_2`.

The attainment is direct: `A - A_k = sum_{i>k} sigma_i u_i v_i^T` has largest singular value
`sigma_{k+1}`.

For the lower bound, let `B` have rank at most `k`, so `null(B)` has dimension at least `n - k`.
Let `W = span{v_1, ..., v_{k+1}}`, of dimension `k + 1`. Since

```
dim null(B) + dim W  >=  (n - k) + (k + 1)  =  n + 1  >  n
```

the two subspaces intersect nontrivially. Take a unit `x` in `null(B) ∩ W`. Then

```
||A - B||_2^2  >=  ||(A - B) x||^2  =  ||A x||^2
```

using `B x = 0`. And since `x` is in `W`, write `x = sum_{i<=k+1} c_i v_i` with `sum c_i^2 = 1`, so

```
||A x||^2 = sum_{i<=k+1} sigma_i^2 c_i^2  >=  sigma_{k+1}^2 sum c_i^2 = sigma_{k+1}^2
```

using that `sigma_i >= sigma_{k+1}` for `i <= k+1`. Therefore `||A - B||_2 >= sigma_{k+1}`.

### 2.2 Prove the Frobenius version, and that the same matrix attains both.

**Claim.** For any `B` of rank at most `k`,
`||A - B||_F^2 >= sum_{i>k} sigma_i^2 = ||A - A_k||_F^2`.

Attainment is again direct.

For the lower bound, use Weyl's inequality for singular values, 41.2.5, in the form

```
sigma_{i+j-1}(X + Y)  <=  sigma_i(X) + sigma_j(Y)
```

Apply it with `X = B`, `Y = A - B`, `j = k + 1`, and `i` running. Since `B` has rank at most `k`,
`sigma_{k+1}(B) = 0`, and more generally `sigma_{i+k}(B) = 0` for `i >= 1`. So

```
sigma_{i+k}(A)  =  sigma_{i+k}(B + (A - B))  <=  sigma_{k+1}(B) + sigma_i(A - B)  =  sigma_i(A - B)
```

Therefore `sigma_i(A - B) >= sigma_{i+k}(A)` for every `i >= 1`, and squaring and summing,

```
||A - B||_F^2 = sum_i sigma_i(A - B)^2  >=  sum_i sigma_{i+k}(A)^2 = sum_{j>k} sigma_j(A)^2
```

**Why the same matrix attains both.** The chain above shows `sigma_i(A - B) >= sigma_{i+k}(A)` for
every `i`, so the entire singular value **sequence** of the error is bounded below, term by term,
by the tail of `A`'s. The truncation achieves equality in every term at once, since `A - A_k` has
singular values exactly `sigma_{k+1}, sigma_{k+2}, ...`.

Any norm that is a monotone function of the singular value sequence is therefore minimised by the
same matrix. The 2-norm takes the first term, the Frobenius norm the sum of squares, and Mirsky's
theorem covers every unitarily invariant norm because all of them are monotone symmetric gauge
functions of that sequence. So the simultaneity is not a coincidence, it is a consequence of the
term-by-term domination.

### 2.3 Show that energy and squared relative error are exactly complementary.

Define

```
E_k = sum_{i<=k} sigma_i^2 / sum_i sigma_i^2       (energy captured)
R_k = ||A - A_k||_F / ||A||_F                       (relative error)
```

By 41.2.4, `||A||_F^2 = sum_i sigma_i^2`, and `||A - A_k||_F^2 = sum_{i>k} sigma_i^2`. So

```
R_k^2  =  sum_{i>k} sigma_i^2 / sum_i sigma_i^2  =  1 - E_k
```

that is, `E_k + R_k^2 = 1` exactly.

**The norm in which this holds is the Frobenius norm, and only that one.** The identity depends on
`||A||_F^2` being the sum of `sigma_i^2`, which is a Pythagorean statement: the rank one pieces
`sigma_i u_i v_i^T` are mutually orthogonal in the Frobenius inner product
`<X, Y> = trace(X^T Y)`, so their squared norms add. The 2-norm has no such additivity, since
`||A||_2 = sigma_1` regardless of the rest, and there is no complementarity there at all.

This is the exact source of the confusion in 43.1.2: the identity is quadratic, so a linear reading
of "98.5 percent of the energy" is wrong by a square root.

### 2.4 Derive the storage condition and find the optimal rank.

A rank `k` approximation of an `m` by `n` matrix stores `U_k` at `mk`, `s` at `k`, and `V_k^T` at
`nk`, so `k(m + n + 1)` numbers against `mn` for the full matrix. Ignoring the `+1`, storing the
factors saves when

```
k (m + n)  <  m n,   that is   k  <  m n / (m + n)
```

For a square matrix that is `k < n/2`, so the SVD saves only for ranks below half. For a `1000` by
`10` matrix it is `k < 9.9`, so essentially only `k = 9` saves anything, and barely.

The **compression ratio** is `mn / (k(m+n+1))`.

**The rank maximising the saving for a given error** is a different question and has a cleaner
answer. Fix a relative Frobenius error target `tau`. By 43.2.3 the smallest admissible rank is

```
k*(tau) = min { k : sum_{i>k} sigma_i^2 <= tau^2 sum_i sigma_i^2 }
```

and since storage is increasing in `k`, that smallest admissible `k` is also the one maximising the
saving. So there is no interior optimum to search for: take the smallest rank meeting the error
target.

```python
def rank_for_error(s, tau):
    """The smallest k whose truncation has relative Frobenius error at most tau."""
    s = np.asarray(s, dtype=float)
    total = float(np.sum(s ** 2))
    tail = total - np.cumsum(s ** 2)
    ok = np.flatnonzero(tail <= tau ** 2 * total)
    return int(ok[0] + 1) if ok.size else s.size
```

The practical consequence is that the interesting question is never "what rank", it is "what error
can I tolerate", and the rank follows from the spectrum. If the spectrum decays fast the answer is
small and compression works; if it decays slowly the answer exceeds `mn/(m+n)` and compression by
truncation is not available at all.

### 2.5 State the Halko-Martinsson-Tropp bound, and identify what `p` and `q` do.

**The bound.** Draw a Gaussian `Omega` of size `n` by `(k + p)` with `p >= 2`, form `Y = A Omega`,
orthonormalise to get `Q`, and set `A_approx = Q Q^T A`. Then

```
E || A - Q Q^T A ||_F  <=  ( 1 + k/(p-1) )^{1/2} * ( sum_{i>k} sigma_i^2 )^{1/2}
```

and in the 2-norm,

```
E || A - Q Q^T A ||_2  <=  ( 1 + sqrt(k/(p-1)) ) sigma_{k+1}
                            + ( e sqrt(k+p) / p ) ( sum_{i>k} sigma_i^2 )^{1/2}
```

With `q` steps of power iteration, replacing `Y = A Omega` by `Y = (A A^T)^q A Omega`, the same
bounds hold with every `sigma_i` replaced by `sigma_i^{2q+1}`, so the effective ratio becomes

```
( sigma_{k+1} / sigma_k )^{2q+1}
```

**What `p` does.** It controls the **constant**. The factor `1 + k/(p-1)` falls rapidly: at
`p = 2` it is `1 + k`, at `p = k` it is about 2, at `p = 2k` it is about 1.5. So `p` buys a
multiplicative constant in front of the optimal error, and its returns diminish sharply once `p` is
comparable to `k`. It cannot make the error smaller than `sigma_{k+1}`, ever.

**What `q` does.** It controls the **exponent**. Each power iteration raises every singular value
to a higher power, which widens the relative gap between what you want and what you do not.
Concretely, if the spectrum decays slowly so `sigma_{k+1}/sigma_k = 0.9`, then `q = 0` leaves a
ratio of 0.9 and `q = 2` gives `0.9^5 = 0.59`. So `q` is the knob that works when the spectrum
decays slowly, and `p` is the knob that works when it decays fast enough that the constant is all
that stands between you and the optimum.

The cost is the difference: `p` costs `p` extra matrix products once, `q` costs a factor of
`2q + 1` on the whole sketch. Exercise 43.4.1 measures where each is the better buy.

### Level 3, computational

### 3.1 A single-pass randomised SVD.

The standard version computes `Q` from `Y = A Omega`, then forms `B = Q^T A`, which touches `A` a
second time. A single-pass version sketches both sides at once and recovers the core from the two
sketches, so nothing after the sketch touches `A`. That matters when `A` arrives as a stream and
cannot be stored.

```python
def randomised_svd_single_pass(A, k, oversample=10, rng=None):
    """Sketch BOTH sides at once: Y = A Om and Z = A^T Psi. Then Q from Y and P from Z, and the
    small core B = Q^T A P is recovered from (Psi^T Q) B = Z^T P as a least squares problem.
    Nothing after the sketch revisits A."""
    gen = np.random.default_rng() if rng is None else rng
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    ell = min(int(k) + int(oversample), min(m, n))
    Om = gen.standard_normal((n, ell))
    Psi = gen.standard_normal((m, ell))
    Y = A @ Om                       # the only two touches of A, and they happen together
    Z = A.T @ Psi
    Q, _ = np.linalg.qr(Y)
    P, _ = np.linalg.qr(Z)
    B = np.linalg.lstsq(Psi.T @ Q, Z.T @ P, rcond=None)[0]
    Ub, s, Vtb = np.linalg.svd(B)
    kk = int(k)
    U = Q @ Ub[:, :kk]
    Vt = (P @ Vtb[:kk].T).T
    return {"matrix": (U * s[:kk]) @ Vt, "U": U, "s": s[:kk], "Vt": Vt, "passes": 1}
```

Measured on a 300 by 200 matrix, `p = 10`:

| spectrum | `k` | optimal `sigma_{k+1}` | two-pass | single-pass | penalty |
|---|---|---|---|---|---|
| fast, `1/i^2` | 10 | 8.264e-3 | 9.165e-3 | 3.620e-1 | 39.5x |
| fast, `1/i^2` | 30 | 1.041e-3 | 1.918e-3 | 1.197e-1 | 62.4x |
| medium, `1/i` | 10 | 9.091e-2 | 1.250e-1 | 4.239e0 | 33.9x |
| medium, `1/i` | 30 | 3.226e-2 | 6.865e-2 | 1.349e0 | 19.7x |
| slow, `i^{-0.5}` | 10 | 3.015e-1 | 4.553e-1 | 1.437e1 | 31.6x |
| slow, `i^{-0.5}` | 30 | 1.796e-1 | 3.421e-1 | 5.074e2 | 1483x |
| flat | 10 | 1.000e0 | 1.000e0 | 2.032e2 | 203x |
| flat | 30 | 1.000e0 | 1.000e0 | 3.878e2 | 388x |

**The second pass is worth a factor of 20 to 1500.** That is the honest measurement and it is much
larger than the literature's typical framing suggests.

The reason is the least squares step. Recovering `B` from `(Psi^T Q) B = Z^T P` requires solving
with `Psi^T Q`, an `ell` by `ell` matrix of random projections, whose condition number is not
controlled and is typically 10 to 100 for these sizes. That conditioning multiplies straight into
the answer. The two-pass version computes `B = Q^T A` exactly, with no solve at all.

So the single-pass version is for the case where a second pass is **impossible**, not merely
expensive: a data stream, a matrix generated on the fly, a distributed matrix too large to reread.
When a second pass is affordable it is worth taking, by a wide margin. Better single-pass schemes
exist, using structured sketches and more careful core recovery, and they narrow the gap without
closing it.

### 3.2 The CUR decomposition.

CUR approximates `A` by **actual columns and rows** of `A` rather than by singular vectors:
`A ≈ C U R` with `C` a subset of columns, `R` a subset of rows, and `U = C^+ A R^+` the choice
making `C U R` the best approximation available from that `C` and `R`.

```python
def cur(A, k, oversample=10, rng=None):
    """Columns and rows are sampled by LEVERAGE SCORE, the row norms of the top k singular
    vectors, which is what makes the sampling near optimal rather than uniform."""
    gen = np.random.default_rng() if rng is None else rng
    A = np.atleast_2d(np.asarray(A, dtype=float))
    m, n = A.shape
    k = int(k)
    c, r = min(k + int(oversample), n), min(k + int(oversample), m)
    U0, _, Vt0 = np.linalg.svd(A, full_matrices=False)
    p_col = np.sum(Vt0[:k] ** 2, axis=0) / k
    p_row = np.sum(U0[:, :k] ** 2, axis=1) / k
    cols = gen.choice(n, size=c, replace=False, p=p_col / p_col.sum())
    rows = gen.choice(m, size=r, replace=False, p=p_row / p_row.sum())
    C, R = A[:, cols], A[rows, :]
    return {"matrix": C @ (np.linalg.pinv(C) @ A @ np.linalg.pinv(R)) @ R,
            "columns": cols, "rows": rows}
```

CUR with target `k` and `p` extra actually returns rank `k + p`, so the honest comparison is
against the SVD truncated to `k + p`, not to `k`:

| spectrum | `k` | CUR rank | SVD at `k` | SVD at CUR's rank | CUR | penalty at same rank |
|---|---|---|---|---|---|---|
| fast, `1/i^2` | 5 | 15 | 2.778e-2 | 3.906e-3 | 1.157e-2 | 2.96x |
| fast, `1/i^2` | 20 | 30 | 2.268e-3 | 1.041e-3 | 4.127e-3 | 3.97x |
| medium, `1/i` | 5 | 15 | 1.667e-1 | 6.250e-2 | 1.686e-1 | 2.70x |
| medium, `1/i` | 20 | 30 | 4.762e-2 | 3.226e-2 | 9.401e-2 | 2.91x |
| slow, `i^{-0.5}` | 5 | 15 | 4.082e-1 | 2.500e-1 | 5.843e-1 | 1.43x |
| slow, `i^{-0.5}` | 20 | 30 | 2.182e-1 | 1.796e-1 | 4.193e-1 | 2.33x |

**CUR costs a factor of 2.3 to 4.0 in error at the same rank.** Note that comparing against the SVD
at `k` rather than at CUR's actual rank makes CUR look better than it is, and in one row it even
appears to beat the SVD. That comparison is wrong and the table above avoids it.

**What CUR buys.** `C` and `R` are made of real columns and rows, so they keep whatever the columns
of `A` meant: a gene, a document, a user, a wavelength. A singular vector is a dense mixture of all
of them and means nothing on its own. For a scientist trying to identify which 20 genes explain the
variation, CUR answers the question and the SVD does not.

Two further advantages. If `A` is **sparse**, `C` and `R` are sparse too, while `U` and `V` from an
SVD are dense; CUR can therefore be far cheaper to store. And CUR needs only the sampled columns
and rows, so `A` need never be fully read once the leverage scores are estimated, which can be done
approximately by a randomised method.

The cost is the factor of 2 to 4, and a `U` that must be computed from a pseudoinverse and can be
ill conditioned.

### 3.3 Sparse PageRank with a personalisation vector.

The Google matrix is dense and `n` by `n`, so for a real web it cannot be formed. Every power
iteration step can still be done in `O(number of edges)`, because the rank one damping term is a
scalar correction and the dangling mass is a single sum.

```python
def sparse_pagerank(edges, n, damping=0.85, personal=None, tol=1e-13, max_iter=10_000):
    """x <- d * (P^T x + dangling_mass / n) + (1 - d) * v, with P^T x a scatter over the edge
    list. Setting v to something other than uniform ranks the graph FROM a chosen node."""
    edges = np.asarray(edges, dtype=np.int64)
    n = int(n)
    src, dst = edges[:, 0], edges[:, 1]
    outdeg = np.bincount(src, minlength=n).astype(float)
    dangling = outdeg == 0
    v = (np.full(n, 1.0 / n) if personal is None
         else np.asarray(personal, dtype=float) / np.sum(personal))
    x = np.full(n, 1.0 / n)
    for step in range(int(max_iter)):
        w = np.zeros(n)
        np.add.at(w, dst, x[src] / outdeg[src])
        new = damping * (w + float(x[dangling].sum()) / n) + (1.0 - damping) * v
        new /= new.sum()
        if float(np.linalg.norm(new - x, 1)) <= tol:
            return {"rank": new, "iterations": step + 1, "converged": True}
        x = new
    return {"rank": x, "iterations": int(max_iter), "converged": False}
```

Measured:

| `n` | edges | iterations | sparse time | dense time | speedup | agreement |
|---|---|---|---|---|---|---|
| 200 | 1932 | 24 | 0.0006s | 0.0006s | 0.9x | 4.0e-15 |
| 2000 | 7983 | 42 | 0.0021s | 0.0571s | 27.2x | 4.1e-15 |
| 20000 | 79991 | 41 | 0.0327s | dense needs 3.2 GB | n/a | n/a |

At `n = 200` the sparse version is no faster, because numpy's dense matrix-vector product is very
fast at that size and `np.add.at` is not. At `n = 2000` it is 27x faster, and at `n = 20000` the
dense Google matrix would need 3.2 gigabytes and the sparse version runs in 33 milliseconds.

**One trap worth naming.** `nalib.pagerank` takes `links[i, j] = 1` to mean "`j` links to `i`", so
an edge list `(src, dst)` fills `links[dst, src]`. Getting that backwards is a silent error: the
reversed graph still gives a plausible looking ranking, and the first version of this measurement
had it wrong and agreed with the dense reference only to 1e-3 instead of 4e-15. A 1e-3 disagreement
between two implementations of the same thing is not roundoff and should always be investigated.

**Personalisation**, on a 500 node graph:

```
personalised on node 0: own rank 0.00182 -> 0.15114 (83x)
    top 5 changes from [111, 313, 209, 394, 492] to [0, 111, 317, 490, 149]
personalised on node 7: own rank 0.00282 -> 0.15193 (54x)
    top 5 changes from [111, 313, 209, 394, 492] to [7, 404, 181, 231, 308]
```

The chosen node's own rank rises by 54x to 83x, and the top five changes almost entirely. That is
the point: personalised PageRank measures importance **relative to a starting point**, which is
what a recommendation needs, and the uniform version measures importance in the abstract.

### Level 4, experimental

### 4.1 Oversampling `p` against power iterations `q`.

Error as a multiple of the optimal `sigma_{k+1}`, at `k = 20`:

Fast decay, `1/i^2`, optimal error 2.268e-3:

| | `q = 0` | `q = 1` | `q = 2` | `q = 4` |
|---|---|---|---|---|
| `p = 0` | 3.61x | 1.32x | 1.21x | 1.03x |
| `p = 5` | 1.93x | 1.01x | 1.00x | 1.00x |
| `p = 10` | 1.40x | 1.00x | 1.00x | 1.00x |
| `p = 20` | 1.03x | 1.00x | 1.00x | 1.00x |

Slow decay, `i^{-0.5}`, optimal error 2.182e-1:

| | `q = 0` | `q = 1` | `q = 2` | `q = 4` |
|---|---|---|---|---|
| `p = 0` | 2.08x | 1.30x | 1.13x | 1.08x |
| `p = 5` | 1.82x | 1.14x | 1.08x | 1.01x |
| `p = 10` | 1.65x | 1.04x | 1.03x | 1.00x |
| `p = 20` | 1.57x | 1.05x | 1.00x | 1.00x |

The cost is `(k + p)` matrix products times `(2q + 1)`, so:

| | `q = 0` | `q = 1` | `q = 2` | `q = 4` |
|---|---|---|---|---|
| `p = 0` | 20 | 60 | 100 | 180 |
| `p = 5` | 25 | 75 | 125 | 225 |
| `p = 10` | 30 | 90 | 150 | 270 |
| `p = 20` | 40 | 120 | 200 | 360 |

**Where each knob is the better buy.**

For **fast decay**, `p` wins. Reaching 1.03x costs 40 products through `p = 20, q = 0`, against 180
products through `p = 0, q = 4`. That is a factor of 4.5 in favour of oversampling. The reason is
43.2.5: with a fast decaying spectrum the ratio `sigma_{k+1}/sigma_k` is already small, so there is
nothing for `q` to improve, and all that stands between you and the optimum is the constant, which
is exactly what `p` controls.

For **slow decay**, `q` wins. `p = 20, q = 0` reaches only 1.57x for 40 products, and no amount of
further oversampling will fix it, since the `p = 20` column has essentially converged. Getting to
1.05x needs `q = 1`, at 90 to 120 products. The ratio `sigma_{k+1}/sigma_k` is near 1 and only
raising it to a power helps.

**The practical rule**: start with `p = 5` to 10 and `q = 0`. If the error is far from
`sigma_{k+1}`, the spectrum is decaying slowly and you need `q`, so add one power iteration at a
time. Adding more `p` past 20 is almost never worth it.

One warning from the tables: at `p = 20, q = 2` in the slow decay case the error is 1.00x while at
`p = 20, q = 1` it is 1.05x and at `p = 5, q = 4` it is 1.01x. These are single draws, not averages,
so differences at the third digit are noise. The pattern across the table is solid; individual
cells are not.

### 4.2 The best denoising rank against the noise level.

A clean rank 8 matrix, 200 by 100, with singular values `linspace(10, 3, 8)`, plus white noise:

| noise | best rank by sweep | true rank | Marchenko-Pastur edge | count above the edge | error at best | error at true rank |
|---|---|---|---|---|---|---|
| 1.0e-3 | 8 | 8 | 0.024 | 8 | 0.0027 | 0.0027 |
| 3.2e-3 | 8 | 8 | 0.076 | 8 | 0.0086 | 0.0086 |
| 1.0e-2 | 8 | 8 | 0.241 | 8 | 0.0271 | 0.0271 |
| 3.2e-2 | 8 | 8 | 0.763 | 8 | 0.0858 | 0.0858 |
| 1.0e-1 | 8 | 8 | 2.414 | 8 | 0.2774 | 0.2774 |
| 3.2e-1 | 4 | 8 | 7.634 | 5 | 0.8064 | 1.0352 |
| 1.0 | 1 | 8 | 24.142 | 0 | 1.9148 | 3.4778 |

**The Marchenko-Pastur prediction is `sigma * (sqrt(m) + sqrt(n))`**, the largest singular value of
a pure noise matrix, and the count above it matches the best rank almost exactly: 8, 8, 8, 8, 8, 5,
0 against 8, 8, 8, 8, 8, 4, 1.

The relationship is not a fitted power law but a **threshold**: the best rank is the number of
signal singular values that stand above the noise floor. As the noise rises, the floor rises past
`sigma_8`, then past `sigma_7`, and the best rank drops one at a time. At noise 0.32 the floor is
7.63 and only 5 of the 8 signal values, which run from 10 down to 3, exceed it. At noise 1.0 the
floor is 24.1 and none of them do.

The last two rows show why the right rank matters. At noise 0.32, truncating at the best rank 4
gives error 0.8064 while truncating at the true rank 8 gives 1.0352, a 28 percent penalty for using
the correct rank of the underlying matrix. **Keeping a signal component that sits below the noise
floor costs you more than discarding it**, because you keep its noise and gain almost none of its
signal.

### 4.3 PageRank iterations against the damping factor.

| damping `d` | iterations | measured `\|lambda_2\|` | bound `d` | ratio | naive prediction |
|---|---|---|---|---|---|
| 0.50 | 20 | 0.243983 | 0.50 | 0.4880 | 39.9 |
| 0.70 | 26 | 0.341576 | 0.70 | 0.4880 | 77.5 |
| 0.85 | 31 | 0.414771 | 0.85 | 0.4880 | 170.0 |
| 0.95 | 35 | 0.463568 | 0.95 | 0.4880 | 538.7 |
| 0.99 | 37 | 0.483087 | 0.99 | 0.4880 | 2749.3 |

**The bound `|lambda_2| <= d` holds in every row**, which is the theorem: the Google matrix is
`d P + (1-d) e v^T`, and the rank one term fixes the top eigenvalue at 1 while every other
eigenvalue of `P` is multiplied by `d`.

The striking column is the ratio, which is **constant at 0.4880** across all five damping values.
That says `|lambda_2| = 0.488 d` exactly for this graph, so the bound is correct but loose by a
factor of about 2, and the missing factor is a property of the link structure `P`, namely its own
second eigenvalue, which is 0.488 here.

The naive prediction `log(tol)/log(d)` overstates badly, by 74x at `d = 0.99`. Using the measured
rate instead, `log(1e-12)/log(0.488 * 0.99) = 37.9`, which matches the measured 37 exactly. So the
iteration count is predicted precisely once you use the true `|lambda_2|` rather than its bound.

The practical reading is that `d = 0.85` is not chosen because larger values are unaffordable. At
`d = 0.99` the cost is only 37 iterations against 31, a 19 percent increase. It is chosen because
`d` controls **how far the ranking looks**: a larger `d` means the random surfer follows more links
before teleporting, so the ranking depends on longer paths and is more sensitive to link farms and
to the graph's fine structure. The choice is about the model, not about the arithmetic.

### Level 5, advanced

### 5.1 Why the SVD is not always the right low rank approximation.

Three settings where a different factorization is preferred.

**One, non-negative data: non-negative matrix factorization.** For a matrix of counts, intensities,
or probabilities, every entry is non-negative and so should the factors be. The SVD produces
singular vectors with mixed signs, so `A ≈ U S V^T` represents the data as differences of
components, which for a document-term matrix means "this topic minus that topic", an
uninterpretable object. NMF constrains `A ≈ W H` with `W, H >= 0`, so every component is an
additive part and every document is a non-negative mixture of topics.

What it gives up: optimality, uniqueness, and a direct algorithm. NMF is NP-hard in general, is
solved by alternating minimisation that converges to a local minimum, gives a different answer from
a different start, and its error at rank `k` is always at least the SVD's.

**Two, interpretability: CUR, as in 43.3.2.** The SVD's factors are dense mixtures of every column.
CUR uses actual columns and rows. What it gives up: a factor of 2 to 4 in error, measured, and a
middle factor `U` that comes from a pseudoinverse and can be ill conditioned.

**Three, streaming: the randomised range finder and frequent directions.** The SVD needs the whole
matrix, several passes over it, and `O(mn)` storage. A streaming setting supplies rows one at a
time and never again. Frequent directions maintains a small sketch, `2k` rows, that is updated per
row in `O(k^2)` and has a deterministic error guarantee. What it gives up: the error is bounded by
`||A - A_k||_F^2 / k` rather than by `sigma_{k+1}`, so it is weaker, and the sketch is not a
factorization of `A` but of an approximation to it.

A fourth worth mentioning: **sparsity**. If `A` is sparse, `U` and `V` are dense, so the truncated
SVD of a sparse matrix can take **more** storage than the original. Sparse PCA constrains the
factors to be sparse, at the cost of optimality and of a much harder optimisation problem.

The pattern across all four: the SVD is optimal in the class of all rank `k` matrices, and every
alternative restricts to a smaller class in exchange for a property the SVD does not have. Since
the class is smaller, the error is worse. There is no free structure.

### 5.2 The optimal hard threshold.

Gavish and Donoho (2014) computed the exactly optimal truncation rank for denoising a matrix with
white noise. For an `m` by `n` matrix with noise level `sigma`, keep every singular value above

```
lambda(beta) * sqrt(max(m, n)) * sigma,     beta = min(m, n) / max(m, n)
```

with

```
lambda(beta) = sqrt( 2(beta + 1) + 8 beta / ( (beta + 1) + sqrt(beta^2 + 14 beta + 1) ) )
```

When the noise level is unknown it is estimated from the **median** singular value, scaled by
`lambda(beta) / sqrt(mu_beta)` where `mu_beta` is the median of the Marchenko-Pastur distribution.

```python
def gavish_donoho_threshold(m, n, noise_level=None, s=None):
    """A formula, not a sweep. The sweep of section 3 needs the CLEAN matrix to compare against,
    which in practice is exactly the thing that is unknown."""
    m, n = int(m), int(n)
    beta = min(m, n) / max(m, n)
    lam = np.sqrt(2 * (beta + 1)
                  + 8 * beta / ((beta + 1) + np.sqrt(beta ** 2 + 14 * beta + 1)))
    if noise_level is not None:
        return lam * np.sqrt(max(m, n)) * float(noise_level)
    lo, hi = (1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2
    grid = np.linspace(lo, hi, 200_001)
    dens = np.sqrt(np.clip((hi - grid) * (grid - lo), 0, None)) / (2 * np.pi * beta * grid)
    cdf = np.cumsum(dens) * (grid[1] - grid[0])
    mu = float(grid[np.searchsorted(cdf, 0.5 * cdf[-1])])
    return (lam / np.sqrt(mu)) * np.median(np.asarray(s, dtype=float))
```

Measured against the sweep of section 3, on the same clean rank 8 matrix, errors relative to
`||clean||_F`:

| noise | sweep `k` | GD, `sigma` known | GD, `sigma` unknown | sweep error | GD error | penalty | no denoising |
|---|---|---|---|---|---|---|---|
| 1.0e-3 | 8 | 8 | 8 | 0.00245 | 0.00245 | 1.000x | 0.00722 |
| 3.2e-3 | 8 | 8 | 8 | 0.00775 | 0.00775 | 1.000x | 0.02282 |
| 1.0e-2 | 8 | 8 | 8 | 0.02451 | 0.02451 | 1.000x | 0.07216 |
| 3.2e-2 | 8 | 8 | 8 | 0.07757 | 0.07757 | 1.000x | 0.22818 |
| 1.0e-1 | 8 | 8 | 8 | 0.25224 | 0.25224 | 1.000x | 0.72158 |
| 3.2e-1 | 4 | 3 | 3 | 0.79126 | 0.81612 | 1.031x | 2.28184 |
| 1.0 | 1 | 0 | 0 | 1.51269 | 1.00000 | **0.661x** | 7.21580 |

**The formula matches the sweep exactly in five of seven rows**, is within 3 percent in the sixth,
and **beats it in the seventh**.

The last row is the interesting one. At noise 1.0 the Gavish-Donoho threshold says keep **nothing**,
rank 0, giving relative error exactly 1.0, meaning the zero matrix. The sweep's best is rank 1 with
error 1.51. The sweep loses because it searches only over ranks 1 and above and cannot express
"keep nothing", while the formula can. When the noise swamps every signal component, the zero
matrix genuinely is the best rank `k` denoiser, and returning it is the right answer.

**The unknown-noise column picked the same rank as the known-noise column in all seven rows.** The
median-based estimator is a strong result: the median singular value of a noisy matrix is dominated
by the noise bulk and barely affected by the handful of signal values, so it estimates `sigma`
robustly without knowing which values are signal.

**Why the formula beats a sweep in practice**, even where their answers agree: the sweep needs the
clean matrix to compute the error it is minimising, and in any real application the clean matrix is
precisely what is unknown. The sweep is only available in a simulation. The formula needs only the
computed singular values and the shape.

### 5.3 Randomised methods beyond the SVD.

**Sketched least squares.** Solve `min ||A x - b||` on a sketch `S A` instead of on `A`. With `S`
an `s` by `m` random matrix, the sketch preserves every norm in the column space of `[A | b]` to
within `1 + eps` once `s` is a few times `n / eps^2`. That is the same subspace embedding property
that makes the randomised range finder work, applied to the residual instead of to the range.

Measured on a dense Gaussian sketch and a CountSketch, which hashes each row into one of `s` buckets
with a random sign and costs `O(nnz)`:

`20000` by `50`, full solve 0.0141s, residual 14.1747:

| `s/n` | Gaussian residual | Gaussian time | CountSketch residual | CountSketch time | speedup |
|---|---|---|---|---|---|
| 5 | 1.1264 | 0.0631s | 1.1281 | 0.0132s | 1.1x |
| 20 | 1.0218 | 0.2403s | 1.0254 | 0.0133s | 1.1x |
| 100 | 1.0052 | 1.3025s | 1.0062 | 0.0175s | 0.8x |

`200000` by `50`, full solve 0.3810s, residual 44.7513:

| `s/n` | Gaussian residual | Gaussian time | CountSketch residual | CountSketch time | speedup |
|---|---|---|---|---|---|
| 5 | 1.1303 | 0.6087s | 1.1438 | 0.1033s | 3.7x |
| 20 | 1.0298 | 2.6822s | 1.0248 | 0.1120s | 3.4x |
| 100 | 1.0047 | 42.1277s | 1.0046 | 0.1707s | 2.2x |

**Two honest findings.** The dense Gaussian sketch is **slower than solving the problem exactly**,
in every row of both tables, up to 110x slower. Forming `S` and computing `S A` costs `O(s m n)`,
which exceeds the `O(m n^2)` of the QR factorization whenever `s > n`, and `s > n` is required for
the sketch to work at all. Dense Gaussian sketching for least squares is a theoretical device, not
a practical one.

CountSketch is practical, and it pays only when `m/n` is large: 1.1x at `20000` by `50`, and 3.7x
at `200000` by `50`, with a residual 14 percent above optimal at `s/n = 5` and 2.5 percent above at
`s/n = 20`.

**Trace estimation.** `trace(A) = E[z^T A z]` for `z` with independent mean zero, unit variance
entries. Each probe costs one matrix-vector product, so the trace of a matrix you can only apply is
reachable without ever forming a diagonal entry.

```python
def hutchinson_trace(matvec, n, n_probes, rng=None):
    """Rademacher probes. The standard error falls like 1/sqrt(n_probes), which is the same
    square root law the range finder pays."""
    gen = np.random.default_rng() if rng is None else rng
    z = gen.integers(0, 2, size=(int(n), int(n_probes))) * 2.0 - 1.0
    vals = np.array([float(z[:, i] @ matvec(z[:, i])) for i in range(int(n_probes))])
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals)))
```

Measured on a 400 by 400 matrix with exact trace 401.5810:

| probes | estimate | error | relative error | reported standard error | `1/sqrt(k)` |
|---|---|---|---|---|---|
| 10 | 425.1917 | 23.6107 | 0.0588 | 15.1019 | 0.3162 |
| 40 | 404.9277 | 3.3467 | 0.0083 | 3.2647 | 0.1581 |
| 160 | 396.8395 | 4.7415 | 0.0118 | 2.3302 | 0.0791 |
| 640 | 400.3704 | 1.2106 | 0.0030 | 1.0703 | 0.0395 |

The reported standard error tracks the actual error closely in three of four rows, and the whole
sequence falls roughly like `1/sqrt(k)`: from 10 to 640 probes, a factor of 64, the error falls
from 23.6 to 1.2, a factor of 19, against the `sqrt(64) = 8` the law predicts. Better than the law
here, and noisy, which is what four samples of a random quantity look like.

**The relation to the range finder's bound.** Both are Monte Carlo estimates and both pay a square
root. Hutchinson's variance falls like `1/k` in the number of probes, so the error falls like
`1/sqrt(k)`. The randomised range finder's error bound in 43.2.5 carries a `sqrt(k/(p-1))` factor
that falls like `1/sqrt(p)` in the oversampling. In both cases the randomness buys you a
dimension-free estimate at the price of a square root convergence rate, and in both cases the fix
for slow convergence is the same: exploit structure. For the range finder that is power iteration,
which raises the singular values to a power. For trace estimation it is Hutch++, which first
removes the top `k` eigenvalues with a randomised range finder and applies Hutchinson only to the
remainder, reducing the variance by the size of the tail and turning `1/sqrt(k)` into `1/k` for
matrices with decaying spectra.

---

## What Part 6 established

Nine lessons, 153 exercises, every computational and experimental answer measured on this
repository rather than quoted.

**Three claims the measurements corrected.**

Deflation error does **not** accumulate with `k`: it is flat at 1e-15 across six deflations in both
a well separated and a clustered spectrum. What grows is the cost, and only at the cluster.

Clustered eigenvalues do **not** degrade the orthogonality of any method: Jacobi, LAPACK and the
dense Schur route all hold at 2e-15 to 6e-15 down to an exactly repeated eigenvalue. What degrades
is the individual eigenvector, which is not a well defined quantity inside a cluster; the invariant
subspace is computed to 1.4e-15 throughout.

The zero-shift SVD sweep is **not** more accurate on column-graded matrices: both variants reach
3e-15 once the deflation criteria are local rather than global, a library fix this measurement
produced. The genuine adversarial case is a single tiny diagonal entry among neighbours of size 1,
where the shifted sweep has relative error 1.49e3 and the zero-shift sweep has 2.01e-13.

**One library defect found and fixed.** `nalib.svdcompute.golub_kahan_svd` compared a diagonal entry
against the largest entry in the active block when testing for a negligible value. On a graded
matrix that discards legitimate values: at spread 1e16, four of thirty entries tripped it and the
relative error was 6.5e-3 instead of 3.3e-15. The local test fixes it, leaves ordinary matrices bit
for bit unchanged in answer and sweep count, and still catches a true zero. All 75 tests in
`tests/test_svdcompute.py` pass after the change.

**The recurring theme.** A guarantee is not an observed difference. Jacobi's `kappa(B)` bound,
Demmel-Kahan's relative accuracy, the Cholesky reduction's independence from `kappa(B)`: each was
measured to hold, and in each case the method without the guarantee performed within one order of
magnitude on ordinary matrices. The guarantee is worth having because it tells you which matrices
are the exception, and testing on random matrices never would.
