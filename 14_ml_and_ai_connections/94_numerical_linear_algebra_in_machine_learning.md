# 94. Numerical Linear Algebra in Machine Learning

**Part 14: Numerical Analysis in Machine Learning and AI**

## Learning objectives

By the end of this lesson you will be able to:

1. Recognise PCA, whitening, ridge regression and low-rank adaptation as objects this course has
   already built, and name the lesson each one comes from.
2. Say why forming a covariance matrix is a numerical mistake, and measure the two digits per digit
   that it costs.
3. Choose between the three routes to a ridge solution on conditioning grounds rather than taste.
4. Tell the three different numbers that all get called "the rank" of a matrix apart, and say which
   question each one answers.
5. Explain what makes a low-rank adapter work, and give the measurement that shows it is not a
   property of weight matrices in general.

## Prerequisites

Lesson 33 (least squares and the squared condition number, which is the whole of section 3).
Lesson 40 (the singular value decomposition). Lesson 41 (truncation, Eckart-Young and randomized
approximation). Lesson 30 (condition numbers). Lesson 05 (cancellation, which turns up in the angle
computation).

---

## 1. Nothing here is new

This part of the course has one job: to show that the numerical analysis already covered is the
numerical analysis inside a machine learning system, and that the failures are the same failures.
This lesson does the linear algebra half.

| What it is called there | What it is here | Lesson |
|---|---|---|
| Principal component analysis | The SVD of a centred data matrix | 40 |
| Explained variance ratio | The squared singular value share | 40 |
| Whitening, ZCA, decorrelation | $X\Sigma^{-1/2}$, defined only up to an orthogonal factor | 39 |
| Ridge regression, weight decay | Regularized least squares, a filter on the spectrum | 33 |
| Low-rank adaptation | A truncated factorization, Eckart-Young | 41 |
| Embedding spectrum, effective rank | The singular values, and three ways to summarise them | 41 |
| Random feature sketch | The randomized range finder | 41 |

The vocabulary is different in each row and the object is the same. That matters practically,
because the numerical warnings this course attached to the right-hand column apply unchanged to the
left-hand one, and they are usually not repeated there.

## 2. PCA is a truncated SVD, and that is the definition

Centre the data, take the SVD, and read off the answers. The directions are the right singular
vectors, the scores are $US$, and the explained variance ratio is the squared singular value share.
Nothing has to be computed twice.

```python
# Standard setup, the same in every lesson of this course.
import sys, pathlib

_root = pathlib.Path.cwd()
while not (_root / "src" / "nalib").is_dir() and _root != _root.parent:
    _root = _root.parent
sys.path.insert(0, str(_root / "src"))

import numpy as np
import matplotlib.pyplot as plt

SEED = 42                                  # fixed so your numbers match the text
rng = np.random.default_rng(SEED)

np.set_printoptions(precision=6, linewidth=100, suppress=False)
plt.rcParams.update({
    "figure.figsize": (7.5, 4.5), "figure.dpi": 110,
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
})
```

```python
from nalib import mllinalg as mla

wanted = np.geomspace(1.0, 1e-3, 6)
data = mla.matrix_with_spectrum(wanted, rows=200, seed=11, centred=True)
model = mla.pca(data)

print(f"data shape {data.shape}, built with a known spectrum")
print(f"{'i':>4}{'wanted':>14}{'recovered':>14}{'relative error':>18}{'explained':>13}")
for i, (want, got) in enumerate(zip(wanted, model['singular_values'])):
    print(f"{i:>4}{want:>14.6e}{got:>14.6e}{abs(got - want) / want:>18.2e}"
          f"{model['explained'][i]:>13.6f}")

share = model["singular_values"] ** 2 / np.sum(model["singular_values"] ** 2)
print(f"\nexplained variance is the squared singular value share: "
      f"{float(np.max(np.abs(model['explained'] - share))):.3e}")
print(f"the directions are orthonormal to "
      f"{float(np.max(np.abs(model['directions'] @ model['directions'].T - np.eye(model['directions'].shape[0])))):.3e}")
print(f"reconstruction from all components returns the data to "
      f"{float(np.max(np.abs(mla.reconstruct(model) - data))):.3e}")

assert np.allclose(model["singular_values"], wanted, rtol=1e-10)
assert np.allclose(mla.reconstruct(model), data, atol=1e-12)
```

*Output:*

```text
data shape (200, 6), built with a known spectrum
   i        wanted     recovered    relative error    explained
   0  1.000000e+00  1.000000e+00          2.22e-16     0.936904
   1  2.511886e-01  2.511886e-01          2.21e-16     0.059115
   2  6.309573e-02  6.309573e-02          4.40e-16     0.003730
   3  1.584893e-02  1.584893e-02          1.09e-15     0.000235
   4  3.981072e-03  3.981072e-03          8.71e-16     0.000015
   5  1.000000e-03  1.000000e-03          0.00e+00     0.000001

explained variance is the squared singular value share: 3.331e-16
the directions are orthonormal to 6.661e-16
reconstruction from all components returns the data to 1.214e-16
```

Three exact statements, all of them recovered to rounding. **The explained variance ratio is not a
statistical quantity that has to be estimated.** It is $\sigma_i^2 / \sum_j \sigma_j^2$, and it is
available the moment the SVD is.

## 3. The covariance matrix is the wrong object to form

Almost every derivation of PCA starts by writing down the covariance matrix
$C = X^{\mathsf T} X / (n-1)$ and diagonalizing it. That is correct mathematics and a numerical
mistake, and it is the *same* mistake lesson 33 made with the normal equations.

The singular values of $X$ are the square roots of the eigenvalues of $C$, so

$$
\kappa(C) = \kappa(X)^2 .
$$

Forming $C$ therefore spends twice as many digits as the problem needs. If $X$ has condition number
$10^8$, then $C$ has condition number $10^{16}$, which in double precision is no condition number at
all.

```python
from nalib import mllinalg as mla

out = mla.the_three_routes_to_pca_agree_until_they_do_not()
print("relative error in the smallest principal value, RMS over 5 matrices")
print(f"{'condition':>12}{'svd':>12}{'covariance':>13}{'gram':>12}"
      f"{'eps*kappa':>12}{'eps*kappa^2':>14}{'negative':>10}")
for row in out["rows"]:
    print(f"{row['condition']:>12.0e}{row['svd_error']:>12.2e}"
          f"{row['covariance_error']:>13.2e}{row['gram_error']:>12.2e}"
          f"{row['predicted_svd']:>12.2e}{row['predicted_covariance']:>14.2e}"
          f"{row['negative_eigenvalues']:>10}")
print(f"\nfitted slope, svd route       : {out['svd_slope']:.4f}")
print(f"fitted slope, covariance route: {out['covariance_slope']:.4f}")

assert out["svd_grows_like_the_condition_number"]
assert out["covariance_grows_like_its_square"]
assert out["some_eigenvalue_goes_negative"]
```

*Output:*

```text
relative error in the smallest principal value, RMS over 5 matrices
   condition         svd   covariance        gram   eps*kappa   eps*kappa^2  negative
       1e+02    1.36e-15     2.81e-13    2.33e-14    2.22e-14      2.22e-12         0
       1e+04    2.72e-14     3.33e-09    1.06e-10    2.22e-12      2.22e-08         0
       1e+06    4.01e-12     1.44e-05    1.35e-06    2.22e-10      2.22e-04         0
       1e+08    2.81e-10     4.99e-01    9.32e-01    2.22e-08      2.22e+00         1
       1e+10    1.85e-08     3.12e+00    1.63e+02    2.22e-06      2.22e+04         6
       1e+12    3.13e-06     2.96e+01    1.51e+04    2.22e-04      2.22e+08        11

fitted slope, svd route       : 0.9450
fitted slope, covariance route: 2.0192
```

**The fitted slopes are $0.945$ and $2.019$.** One digit lost per digit of conditioning for the SVD
route, two for the covariance route, which is the theory read straight off a log-log fit.

The last column is worse than the accuracy loss. From a data condition number of $10^8$ the
covariance matrix, which is positive definite by construction, comes back with a **negative**
eigenvalue. Its square root is not a real number, so the code does not return an inaccurate answer:
it returns something that is not an answer.

Note also that the Gram route, $XX^{\mathsf T}$ instead of $X^{\mathsf T}X$, is not a rescue. It is
worse here, $0.93$ against $0.499$ at a condition number of $10^8$, because it hands a $200 \times
200$ eigensolver the job of separating $12$ tiny eigenvalues from $188$ exact zeros. **The Gram
route is a cost optimization for the case where features outnumber samples, and it is not a
stability one.**

## 4. PCA reads whatever units you gave it

The covariance matrix depends on the scale of each feature, so PCA does too. This is not a numerical
problem, it is a modelling one, and it is worth measuring because the size of the effect surprises
people.

```python
from nalib import mllinalg as mla

out = mla.pca_is_not_scale_invariant()
print("multiply feature 0 by a constant and look at the leading principal direction")
print(f"{'factor':>10}{'angle moved':>14}{'explained':>12}{'weight on it':>15}"
      f"{'whitened geometry':>20}")
for row in out["rows"]:
    print(f"{row['factor']:>10.0f}{row['angle']:>14.4f}{row['explained']:>12.6f}"
          f"{row['leading_weight_on_the_scaled_feature']:>15.6f}"
          f"{row['whitened_gram_change']:>20.2e}")

assert out["directions_move"]
assert out["the_scaled_feature_takes_over"]
assert out["whitening_is_invariant"]
```

*Output:*

```text
multiply feature 0 by a constant and look at the leading principal direction
    factor   angle moved   explained   weight on it   whitened geometry
         1        0.0000    0.408241       0.701463            0.00e+00
        10       42.4898    0.976036       0.998655            6.05e-15
       100       45.1608    0.999754       0.999987            3.63e-14
      1000       45.4260    0.999998       1.000000            1.74e-13
```

A factor of $1000$, which is the difference between metres and millimetres, turns the leading
direction through $45.4$ degrees and gives that one feature all of the weight. **PCA has found the
unit, not the structure.**

The last column is the contrast. The whitened sample geometry, measured as the Gram matrix
$ZZ^{\mathsf T}$ of the whitened data, does not move at all: $1.7 \times 10^{-13}$. Whitening
divides every direction by its own standard deviation, so it cannot see the units. That is the
practical argument for standardizing features before PCA, stated as a measurement.

## 5. Whitening is not unique, and the choice is not statistical

Whitening means finding $W$ with $\operatorname{cov}(XW) = I$. If $W$ works then so does $WQ$ for
any orthogonal $Q$, because $Q^{\mathsf T} I Q = I$. So there are infinitely many whiteners and no
statistical criterion picks between them.

Two get names. **PCA whitening** rotates to the principal axes and rescales, $W = V\Lambda^{-1/2}$.
**ZCA whitening** rotates back afterwards, $W = V\Lambda^{-1/2}V^{\mathsf T}$, which is the
symmetric square root of $C^{-1}$.

```python
from nalib import mllinalg as mla

out = mla.whitening_is_not_unique()
print(f"covariance error, pca whitening: {out['pca_covariance_error']:.3e}")
print(f"covariance error, zca whitening: {out['zca_covariance_error']:.3e}")
print(f"the two differ by a factor that is orthogonal to "
      f"{out['rotation_orthogonality']:.3e}")
print(f"\ndistance from the identity map, pca: {out['pca_distance']:.4f}")
print(f"distance from the identity map, zca: {out['zca_distance']:.4f}")
print(f"largest amplification over smallest: {out['amplification_ratio']:.4f}")

assert out["both_whiten"]
assert out["they_differ_by_a_rotation"]
assert out["zca_is_closer_to_the_identity"]
```

*Output:*

```text
covariance error, pca whitening: 1.110e-15
covariance error, zca whitening: 1.554e-15
the two differ by a factor that is orthogonal to 2.220e-15

distance from the identity map, pca: 8.9970
distance from the identity map, zca: 5.8616
largest amplification over smallest: 4.2597
```

Both whiten exactly, to $1.6\times10^{-15}$, and the map between them is orthogonal to
$2.2\times10^{-15}$. **The choice is a choice about what to preserve.** ZCA is the whitener closest
to the identity, at $5.86$ against PCA whitening's $9.00$, which is why image work uses it: it
leaves the picture looking like a picture.

## 6. Whitening on data you have not seen

Every whitener is exact on its own sample. That is what "fit" means here, and it is why the training
number carries no information at all. The question is what happens to fresh data.

```python
from nalib import mllinalg as mla

out = mla.whitening_needs_a_penalty_to_generalize()
print("fit on 3 samples per dimension, then apply to fresh data from the same distribution")
print(f"{'penalty':>12}{'amplification':>16}{'on its own sample':>20}{'on fresh data':>16}")
for row in out["rows"]:
    print(f"{row['penalty']:>12.0e}{row['amplification']:>16.2f}"
          f"{row['training_error']:>20.2e}{row['held_out_error']:>16.4f}")
print(f"\nbest penalty {out['best_penalty']:.0e}, which cuts the held out error by "
      f"{out['gain']:.4f}")

assert out["training_error_says_nothing"]
assert out["the_penalty_helps"]
assert out["too_much_penalty_hurts"]
```

*Output:*

```text
fit on 3 samples per dimension, then apply to fresh data from the same distribution
     penalty   amplification   on its own sample   on fresh data
       0e+00          505.29            2.38e-14          1.3370
       1e-06          450.98            9.60e-02          1.1820
       1e-04           98.10            7.85e-01          0.7025
       1e-03           31.56            9.06e-01          0.8946
       1e-02           10.00            9.47e-01          0.9517
       1e-01            3.16            9.79e-01          0.9792

best penalty 1e-04, which cuts the held out error by 1.9033
```

The unregularized whitener is exact on its own sample, to $2.4\times10^{-14}$, and its covariance on
fresh data is off by $1.34$. A penalty of $10^{-4}$ cuts that to $0.70$.

The mechanism is worth stating plainly, because it is the same one that will appear three more times
in this part. **The smallest sample direction is the one estimated worst, and whitening divides by
it.** An unregularized whitener trusts its weakest measurement most. The fix, $1/\sqrt{\lambda_i +
\alpha}$, is ridge regression wearing a different name, and section 7 is about the name it usually
wears.

## 7. Ridge regression is a filter on the spectrum

Ridge solves

$$
\min_x \; \lVert Ax - b\rVert_2^2 + \lambda \lVert x \rVert_2^2 .
$$

Write $A = U\Sigma V^{\mathsf T}$ and the solution falls out as

$$
x_\lambda \;=\; \sum_i \frac{\sigma_i}{\sigma_i^2 + \lambda}\,(u_i^{\mathsf T}b)\, v_i
\;=\; \sum_i f_i(\lambda) \, \frac{u_i^{\mathsf T}b}{\sigma_i} \, v_i ,
\qquad
f_i(\lambda) = \frac{\sigma_i^2}{\sigma_i^2 + \lambda} .
$$

The second form is the useful one. Ridge takes the plain least squares solution and multiplies
component $i$ by $f_i$, a number between $0$ and $1$. Directions with $\sigma_i^2 \gg \lambda$ are
untouched. Directions with $\sigma_i^2 \ll \lambda$ are switched off. The crossover is exactly at
$\sigma_i^2 = \lambda$.

```python
from nalib import mllinalg as mla

out = mla.ridge_is_a_spectral_filter()
print(f"singular values from {out['singular_values'][0]:.3e} down to "
      f"{out['singular_values'][-1]:.3e}")
print(f"{'penalty':>10}{'filter error':>15}{'largest f':>12}{'smallest f':>13}"
      f"{'condition':>13}{'predicted':>13}{'kept':>7}")
for row in out["rows"]:
    print(f"{row['penalty']:>10.0e}{row['filter_error']:>15.2e}{row['largest_filter']:>12.6f}"
          f"{row['smallest_filter']:>13.2e}{row['condition']:>13.4e}"
          f"{row['predicted_condition']:>13.4e}{row['kept']:>7}")

assert out["filters_are_exact"]
assert out["condition_formula_holds"]
```

*Output:*

```text
singular values from 1.000e+00 down to 1.000e-06
   penalty   filter error   largest f   smallest f    condition    predicted   kept
     0e+00       6.66e-16    1.000000     1.00e+00   1.0000e+12   1.0000e+12     12
     1e-08       1.86e-15    1.000000     1.00e-04   9.9990e+07   9.9990e+07      8
     1e-04       2.29e-15    0.999900     1.00e-08   1.0001e+04   1.0001e+04      4
     1e-02       1.78e-15    0.990099     1.00e-10   1.0100e+02   1.0100e+02      2
     1e+00       1.11e-15    0.500000     1.00e-12   2.0000e+00   2.0000e+00      0
```

The filter factors are exact to $2.3\times10^{-15}$, and the condition number of the ridge problem,

$$
\kappa_\lambda \;=\; \frac{\sigma_{\max}^2 + \lambda}{\sigma_{\min}^2 + \lambda} ,
$$

matches to rounding. The "kept" column counts the directions with $f_i > \tfrac12$, and it falls
from $12$ to $0$ as the penalty rises. **Ridge does not shrink the solution evenly. It shrinks the
directions the data did not measure**, and that is why it fixes conditioning and why it introduces
bias, in one line.

## 8. Three ways to solve it, and only one of them is safe

The filter formula is the definition. It is not how anyone computes. The three practical routes are:

```text
normal equations   solve (A'A + lam I) x = A'b               d by d, cheapest
augmented          least squares on [A ; sqrt(lam) I]        stable
svd                apply the filter factors directly          most information, most work
```

The middle one deserves attention. Stack the identity under $A$:

$$
\tilde A = \begin{bmatrix} A \\ \sqrt{\lambda}\, I \end{bmatrix},
\qquad
\tilde b = \begin{bmatrix} b \\ 0 \end{bmatrix} .
$$

Then $\tilde A^{\mathsf T}\tilde A = A^{\mathsf T}A + \lambda I$, so the ordinary least squares
problem $\min \lVert \tilde A x - \tilde b\rVert$ **is** the ridge problem. And the singular values
of $\tilde A$ are $\sqrt{\sigma_i^2 + \lambda}$, so its condition number is the square root of the
normal equations matrix's. Same answer, half the digits spent.

```python
from nalib import mllinalg as mla

out = mla.the_augmented_form_is_the_stable_one()
print(f"{'penalty':>10}{'normal eq error':>18}{'augmented error':>18}"
      f"{'kappa(gram)':>14}{'kappa(stacked)':>16}")
for row in out["rows"]:
    print(f"{row['penalty']:>10.0e}{row['normal_error']:>18.3e}"
          f"{row['augmented_error']:>18.3e}{row['normal_condition']:>14.3e}"
          f"{row['augmented_condition']:>16.3e}")
print(f"\nworst gap {out['worst_gap']:.0f} at penalty {out['worst_penalty']:.0e}, "
      f"which is {out['digits_lost']:.2f} digits")
print(f"kappa(gram) = kappa(stacked)^2 to {out['worst_condition_mismatch']:.1e}")

assert out["augmented_is_better"]
assert out["condition_is_squared"]
```

*Output:*

```text
   penalty   normal eq error   augmented error   kappa(gram)  kappa(stacked)
     1e-02         6.151e-15         3.154e-15     1.010e+02       1.005e+01
     1e-04         1.479e-13         3.863e-15     1.000e+04       1.000e+02
     1e-06         3.064e-11         1.299e-13     1.000e+06       1.000e+03
     1e-08         2.490e-09         3.300e-12     1.000e+08       1.000e+04
     1e-10         3.059e-07         1.214e-11     9.999e+09       1.000e+05
     1e-12         8.028e-06         1.043e-10     9.901e+11       9.950e+05

worst gap 76988 at penalty 1e-12, which is 4.89 digits
kappa(gram) = kappa(stacked)^2 to 2.3e-06
```

**At $\lambda = 10^{-12}$ the normal equations route is $76988$ times less accurate**, which is
$4.89$ digits, and the two condition numbers satisfy $\kappa_{\text{gram}} =
\kappa_{\text{stacked}}^2$ to $2.3\times10^{-6}$.

That last residual is itself informative. It is not a fitting error: it is the rounding committed in
forming $A^{\mathsf T}A$. At $\lambda = 10^{-12}$ the shifted smallest eigenvalue is about
$10^{-12}$, and the rounding in the product is about $\varepsilon\lVert A\rVert^2 \approx
2\times10^{-16}$, so the small eigenvalue of the matrix you actually built is a couple of parts in
$10^6$ away from the one you meant to build. **The Gram matrix does not merely lose accuracy while
solving. It is the wrong matrix before the solve begins.**

## 9. Ridge answers noise, not conditioning

The standard picture of regularization is a U-shaped error curve: too little penalty and the
variance is large, too much and the bias is. It is worth checking when that picture is true.

```python
from nalib import mllinalg as mla

out = mla.ridge_does_not_always_help()
print(f"penalty grid: 0 and {out['penalties'][1]:.0e} to {out['penalties'][-1]:.0e}")
print(f"{'noise':>10}{'best penalty':>16}{'error there':>15}{'error at 0':>15}{'gain':>13}")
for row in out["rows"]:
    print(f"{row['noise']:>10.0e}{row['best_penalty']:>16.3e}{row['best_error']:>15.4e}"
          f"{row['error_at_zero']:>15.4e}{row['gain']:>13.4f}")

assert out["no_noise_wants_no_penalty"]
assert out["noise_wants_a_penalty"]
assert out["best_penalty_grows_with_noise"]
```

*Output:*

```text
penalty grid: 0 and 1e-14 to 1e+02
     noise    best penalty    error there     error at 0         gain
     0e+00       0.000e+00     3.2186e-12     3.2186e-12       1.0000
     1e-06       0.000e+00     1.3611e-01     1.3611e-01       1.0000
     1e-03       3.162e-07     4.4340e+00     3.4795e+01       7.8473
     1e-01       3.162e-03     5.3870e+00     7.4730e+03    1387.2213
```

**With no noise the best penalty is exactly zero**, on a problem whose condition number is $10^5$.
Ill-conditioning on its own is not a reason to regularize: if the right-hand side is exactly in the
range of $A$, the ill-conditioned solve gives the right answer and any penalty moves away from it.

At a noise level of $10^{-6}$ the best penalty is **still** zero, at a gain of $1.0000$. Only at
$10^{-3}$ does a penalty start to pay, and by $10^{-1}$ it is worth a factor of $1387$.

So the U-shaped curve is a picture of the noisy case, and the quantity that decides where the
minimum sits is the ratio of the noise to the smallest singular value, not the condition number by
itself. **Ridge trades bias for variance, and with no variance there is nothing to trade.**

## 10. Rank is three different numbers

An embedding table, a kernel matrix and an attention matrix all tend to have a spectrum that decays
like a power law, $\sigma_i \sim i^{-a}$. Such a matrix is full rank and behaves as though it were
not, and the literature uses at least three summaries without always saying which:

$$
\text{numerical rank} = \#\{\sigma_i > \tau\}, \qquad
\text{stable rank} = \frac{\lVert A\rVert_F^2}{\lVert A\rVert_2^2}, \qquad
\text{effective rank} = e^{H(p)}, \; p_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2} .
$$

```python
from nalib import mllinalg as mla

out = mla.a_decaying_spectrum_has_no_single_rank()
print(f"a {out['size']} by {out['size']} matrix with sigma_i ~ i^-a")
print(f"{'a':>6}{'numerical':>12}{'effective':>12}{'stable':>10}{'90% of energy':>16}"
      f"{'energy in top 1/16':>21}")
for row in out["rows"]:
    print(f"{row['decay']:>6.1f}{row['numerical_rank']:>12}{row['effective_rank']:>12.4f}"
          f"{row['stable_rank']:>10.4f}{row['rank_for_ninety_percent']:>16}"
          f"{row['energy_in_the_top_sixteenth']:>21.6f}")

assert out["all_are_full_rank"]
assert out["the_three_ranks_disagree"]
```

*Output:*

```text
a 256 by 256 matrix with sigma_i ~ i^-a
     a   numerical   effective    stable   90% of energy   energy in top 1/16
   0.5         256     74.6298    6.1243             139             0.552015
   1.0         256      4.9871    1.6410               6             0.965455
   1.5         256      1.9707    1.2020               2             0.998480
   2.0         256      1.3962    1.0823               1             0.999932
   3.0         256      1.0975    1.0173               1             1.000000
```

At $a = 1$ the same matrix has numerical rank $256$, effective rank $4.99$ and stable rank $1.64$.
The spread reaches $252$. **Full rank and effectively low rank are not in conflict**, because they
answer different questions: can this matrix be inverted, and how many directions carry the energy.

The practical reading is the last column. At $a = 1$, the top sixteenth of the directions carry
$96.5$ per cent of the energy, and at $a = 1.5$ they carry $99.8$ per cent. That is the observation
compression and low-rank adaptation are built on.

## 11. Low-rank adaptation, and why it works

A low-rank adapter freezes a weight matrix $W_0$ and learns an update $BA$ with $B$ of shape
$(m, r)$ and $A$ of shape $(r, n)$. It costs $r(m+n)$ parameters instead of $mn$.

The usual justification, "weight updates are low rank", is worth being suspicious of. Here is a
sharper statement, and it is exact. One minibatch gradient of a linear layer is

$$
\nabla_W \mathcal{L} = \frac{1}{B}\, X_{\text{batch}}^{\mathsf T} R ,
$$

a sum of $B$ outer products, so it has rank at most $B$. After $k$ steps the accumulated update has
rank at most $kB$, **whatever the size of the weight matrix**. The bound has no assumption about the
data in it at all.

```python
from nalib import mllinalg as mla

out = mla.a_trained_update_is_low_rank_and_a_random_one_is_not()
print(f"a 128 by 128 layer, batch size 1, rank {out['adapter_rank']} adapter")
print(f"{'steps':>8}{'rank bound':>13}{'measured rank':>16}{'captured by the adapter':>26}")
for row in out["rows"]:
    print(f"{row['steps']:>8}{row['bound']:>13}{row['numerical_rank']:>16}"
          f"{row['captured_by_the_adapter']:>26.6f}")
print(f"\nthe same adapter on a random matrix of the same shape: "
      f"{out['random_captured']:.6f}")
print(f"a flat spectrum would give                             : "
      f"{out['flat_spectrum_share']:.6f}")
print(f"random matrix numerical rank: {out['random_numerical_rank']}")
print(f"parameters saved: a factor of {out['parameter_ratio']:.1f}")

assert out["the_bound_is_exact"]
assert out["the_adapter_captures_what_fits"]
assert out["it_does_not_capture_noise"]
```

*Output:*

```text
a 128 by 128 layer, batch size 1, rank 8 adapter

   steps   rank bound   measured rank   captured by the adapter
       1            1               1                  1.000000
       2            2               2                  1.000000
       4            4               4                  1.000000
       8            8               8                  1.000000
      16           16              16                  0.980565

the same adapter on a random matrix of the same shape: 0.210678
a flat spectrum would give                             : 0.062500
random matrix numerical rank: 128
parameters saved: a factor of 8.0
```

**The bound is not just a bound, it is attained exactly**, at every step count tested. A rank $8$
adapter captures $100$ per cent of an $8$-step update and $98.06$ per cent of a $16$-step one.

The control is the point. The same adapter applied to a random matrix of the same shape captures
$21.07$ per cent. So the claim is not "matrices of this size are low rank", it is **"the path the
optimizer takes is low rank"**, and those are different claims with different consequences. A
long training run with a large batch has a much weaker bound than a short one with a small batch,
and the measurement says so.

One more honest detail: $21.07$ per cent is well above the $6.25$ per cent a flat spectrum would
give. Even a Gaussian matrix has a decaying spectrum, because that is what the Marchenko-Pastur law
says. A rank $8$ adapter on random data is not useless, it is just five times worse than on a
trained update.

## 12. Nothing beats the truncated SVD

Eckart-Young from lesson 41 says the truncated SVD is exactly the best rank-$r$ approximation in
both the spectral and Frobenius norms. So the only open question about the cheaper alternatives is
how much they give up.

```python
from nalib import mllinalg as mla

out = mla.nothing_beats_the_truncated_svd()
print(f"{'rank':>6}{'optimal':>12}{'random':>12}{'oversampled':>14}{'columns':>11}"
      f"{'random/opt':>13}{'over/opt':>11}{'cols/opt':>11}")
for row in out["rows"]:
    print(f"{row['rank']:>6}{row['optimal']:>12.6f}{row['random']:>12.6f}"
          f"{row['oversampled']:>14.6f}{row['columns']:>11.6f}"
          f"{row['random_ratio']:>13.4f}{row['oversampled_ratio']:>11.4f}"
          f"{row['column_ratio']:>11.4f}")

assert out["optimal_is_optimal"]
assert out["oversampling_helps"]
```

*Output:*

```text
  rank     optimal      random   oversampled    columns   random/opt   over/opt   cols/opt
     2    0.486043    0.687673      0.606810   0.522123       1.4148     1.2485     1.0742
     4    0.360777    0.568072      0.418211   0.415522       1.5746     1.1592     1.1517
     8    0.258322    0.373651      0.308941   0.334252       1.4465     1.1960     1.2939
    16    0.178743    0.276516      0.218860   0.295682       1.5470     1.2244     1.6542
    32    0.117168    0.193271      0.136776   0.282132       1.6495     1.1674     2.4079
```

Nothing beats it, at any rank. A randomized range finder with no oversampling loses up to $1.65$; add
$r$ extra columns and that falls to $1.25$. A column subset, which has the advantage of being made of
real features you can name, loses $2.41$ at rank $32$ and gets worse as the rank grows, because the
greedy choice runs out of good columns.

## 13. The picture

```python
from nalib import mllinalg as mla

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

routes = mla.the_three_routes_to_pca_agree_until_they_do_not()
conditions = [r["condition"] for r in routes["rows"]]
axes[0].loglog(conditions, [r["svd_error"] for r in routes["rows"]], "o-", label="svd")
axes[0].loglog(conditions, [r["covariance_error"] for r in routes["rows"]], "s-",
               label="covariance")
axes[0].loglog(conditions, [r["predicted_svd"] for r in routes["rows"]], ":", color="0.5",
               label="eps kappa")
axes[0].loglog(conditions, [r["predicted_covariance"] for r in routes["rows"]], "-.",
               color="0.5", label="eps kappa^2")
axes[0].set_xlabel("condition number of the data")
axes[0].set_ylabel("relative error")
axes[0].set_title("two slopes, not one")
axes[0].legend(fontsize=7)

values = np.geomspace(1.0, 1e-6, 60)
for penalty, style in ((1e-10, "-"), (1e-6, "--"), (1e-2, "-.")):
    axes[1].loglog(values, mla.ridge_filters(values, penalty), style,
                   label=f"lam = {penalty:.0e}")
axes[1].set_xlabel("singular value")
axes[1].set_ylabel("filter factor")
axes[1].set_title("ridge switches directions off")
axes[1].legend(fontsize=7)

for decay, style in ((0.5, "-"), (1.0, "--"), (2.0, "-.")):
    line = mla.spectrum(128, "power", decay)
    axes[2].loglog(np.arange(1, line.size + 1), line, style, label=f"a = {decay}")
trained = mla.train_linear_layer(128, 128, 512, 8, 1, seed=23)["singular_values"]
axes[2].loglog(np.arange(1, trained.size + 1), trained / trained[0], "k:",
               label="an 8-step update")
axes[2].set_ylim(1e-17, 2.0)
axes[2].set_xlabel("index")
axes[2].set_ylabel("singular value")
axes[2].set_title("power law against a rank 8 update")
axes[2].legend(fontsize=7)

fig.tight_layout(); fig.savefig("../figures/94_mllinalg.png", dpi=110); plt.close(fig)
print("saved ../figures/94_mllinalg.png")
```

*Output:*

```text
saved ../figures/94_mllinalg.png
```

![Conditioning of the three PCA routes, the ridge filter factors, and a power-law spectrum against a trained update](../figures/94_mllinalg.png)

The left panel is section 3: two straight lines of visibly different slope, each tracking its
predicted bound. The middle is section 7: ridge is a soft cutoff at $\sigma^2 = \lambda$. The right
is sections 10 and 11 on the same axes, and the difference is what a rank bound looks like next to a
power law. The power laws never reach zero. The trained update falls off a cliff at index $8$.

## 14. From scratch

PCA, whitening and ridge are each about five lines once you have the SVD. Writing them out is worth
doing once, because the shared structure is invisible in the library forms.

```python
def my_pca(data, components):
    """Centre, decompose, truncate."""
    mean = data.mean(axis=0)
    left, s, right = np.linalg.svd(data - mean, full_matrices=False)
    return mean, right[:components], left[:, :components] * s[:components], s


def my_whiten(data, penalty=0.0, symmetric=True):
    """Same decomposition, different use of the same three factors."""
    mean = data.mean(axis=0)
    _, s, right = np.linalg.svd(data - mean, full_matrices=False)
    variance = s ** 2 / (data.shape[0] - 1)
    scale = 1.0 / np.sqrt(variance + penalty)
    matrix = right.T * scale
    return (matrix @ right) if symmetric else matrix


def my_ridge(matrix, target, penalty):
    """The stacked form, so the condition number stays square-rooted."""
    stacked = np.vstack([matrix, np.sqrt(penalty) * np.eye(matrix.shape[1])])
    padded = np.concatenate([target, np.zeros(matrix.shape[1])])
    return np.linalg.lstsq(stacked, padded, rcond=None)[0]


from nalib import mllinalg as mla

data = mla.dataset(150, 8, decay=0.8, seed=77)["data"]
rng_local = np.random.default_rng(0)
target = rng_local.standard_normal(data.shape[0])

mean, directions, scores, values = my_pca(data, 3)
library = mla.pca(data, components=3)
print(f"pca directions agree to        "
      f"{float(np.max(np.abs(np.abs(directions) - np.abs(library['directions'])))):.3e}")

whitened = (data - data.mean(axis=0)) @ my_whiten(data)
covariance = whitened.T @ whitened / (data.shape[0] - 1)
print(f"my whitener gives the identity to "
      f"{float(np.max(np.abs(covariance - np.eye(data.shape[1])))):.3e}")

print(f"my ridge agrees with the filter form to "
      f"{float(np.max(np.abs(my_ridge(data, target, 1e-3) - mla.ridge_svd(data, target, 1e-3)['solution']))):.3e}")

assert np.allclose(np.abs(directions), np.abs(library["directions"]), atol=1e-10)
assert np.allclose(my_ridge(data, target, 1e-3), mla.ridge_svd(data, target, 1e-3)["solution"],
                   atol=1e-8)
```

*Output:*

```text
pca directions agree to        0.000e+00
my whitener gives the identity to 1.998e-15
my ridge agrees with the filter form to 5.551e-16
```

Three functions, one decomposition. **The SVD is the whole of this lesson**, and the machine learning
names are three different ways of using its three factors.

## 15. Exercises

**Level 1, understanding**

1.1 State PCA in terms of the SVD, without mentioning covariance, and say what the explained
variance ratio is in that language.

1.2 Explain why $\kappa(X^{\mathsf T}X) = \kappa(X)^2$ and what that costs in digits.

1.3 Say why whitening is not unique, and name two whiteners and what each one preserves.

1.4 Write down the ridge filter factor and say what it does to a direction with
$\sigma^2 \gg \lambda$ and to one with $\sigma^2 \ll \lambda$.

1.5 Give the rank bound on an accumulated minibatch update and say what it does and does not assume.

**Level 2, derivation**

2.1 Derive the ridge solution from the SVD and show it reduces to the pseudoinverse at
$\lambda \to 0$.

2.2 Show that the singular values of $[A; \sqrt\lambda I]$ are $\sqrt{\sigma_i^2 + \lambda}$, and
deduce the condition number relation used in section 8.

2.3 Show that ZCA whitening is the whitener that minimizes $\lVert W - I\rVert_F$ over all
whiteners, using the orthogonal Procrustes result.

2.4 Derive the bias and variance of the ridge estimator in the SVD basis and locate the minimum of
their sum.

2.5 Prove that a sum of $k$ rank-one updates has rank at most $k$, and give the condition for
equality.

**Level 3, computational**

3.1 Implement PCA three ways, including one that never forms a matrix larger than $n \times r$, and
compare accuracy and cost as the condition number grows.

3.2 Implement generalized cross-validation for choosing $\lambda$ and compare its choice against the
oracle penalty that minimizes the true error.

3.3 Implement kernel PCA with the double-centring trick, and say which of the three routes of
section 3 it corresponds to.

3.4 Implement a rank-$r$ adapter for a two-layer network, train it, and measure how much of the full
fine-tuning update it recovers as $r$ varies.

3.5 Implement incremental PCA that processes the data in blocks, and measure its drift against the
batch answer over many blocks.

**Level 4, experimental**

4.1 Measure the crossover in cost between the covariance route and the SVD route as the ratio of
samples to features changes, and find where each one wins.

4.2 Measure how the effective rank of a weight matrix changes over a training run, and say whether
the change is in the weights or in the update.

4.3 Measure the penalty that generalized cross-validation picks as the noise level varies over six
decades, and compare its shape against the oracle from section 9.

**Level 5, advanced**

5.1 **Why the covariance matrix survives in practice.** Given section 3, explain why the covariance
route is still the default in most statistics packages, and state the condition under which it is a
defensible choice.

5.2 **Regularization by truncation against regularization by ridge.** Compare the truncated SVD
filter, which is $0$ or $1$, against the ridge filter, which is smooth, and say when the difference
matters.

5.3 **What a low-rank adapter cannot do.** Given that the update has rank at most $kB$, describe a
training situation in which a rank-$r$ adapter must lose information, and say what the loss looks
like.

## 16. Key takeaways

- **PCA is the SVD of a centred data matrix**, recovered here to $10^{-10}$ relative on a known
  spectrum, and the explained variance ratio is $\sigma_i^2/\sum\sigma_j^2$ exactly.

- **The covariance route squares the condition number.** The measured slopes are $0.945$ for the SVD
  route and $2.019$ for the covariance route, one digit lost per digit of conditioning against two.

- **Past a condition number of $10^8$ the covariance route stops being defined**, returning a
  negative eigenvalue for a positive definite matrix.

- **The Gram route is a cost optimization and not a stability one.** It is worse than the covariance
  route here, $0.93$ against $0.499$.

- **PCA reads units.** A factor of $1000$ on one feature moves the leading direction by $45.4$
  degrees, while the whitened geometry does not move at all: $1.7\times10^{-13}$.

- **Whitening is not unique**, and PCA and ZCA whitening differ by an exact rotation, orthogonal to
  $2.2\times10^{-15}$. ZCA is the one closest to the identity, $5.86$ against $9.00$.

- **A whitener is exact on its own sample and wrong on fresh data.** Held-out error $1.34$ at
  $\lambda = 0$ against $0.70$ at $\lambda = 10^{-4}$.

- **Ridge is a filter, $f_i = \sigma_i^2/(\sigma_i^2+\lambda)$**, exact to $2.3\times10^{-15}$, and
  the condition number formula is exact to rounding.

- **The augmented form is the one to compute with.** At $\lambda = 10^{-12}$ it is $76988$ times
  more accurate, which is $4.89$ digits.

- **Ridge answers noise, not conditioning.** On a problem with condition number $10^5$ and no noise
  the best penalty is exactly zero, and at a noise level of $10^{-6}$ it is still zero.

- **Rank is three numbers.** A $256\times256$ matrix with $\sigma_i \sim 1/i$ has numerical rank
  $256$, effective rank $4.99$ and stable rank $1.64$.

- **A trained update has rank exactly $\min(kB, d)$**, attained at every step count tested, so a rank
  $8$ adapter captures $100$ per cent of an $8$-step update and $21.07$ per cent of a random matrix.
  Low rank is a property of the optimizer's path.

- **Nothing beats the truncated SVD**, and the alternatives lose by $1.65$ (random projection),
  $1.25$ (with oversampling) and $2.41$ (column subset).

## Where this goes next

Lesson 95 takes the other half of the same system, the optimizer, and asks the same question of it:
which of Part 12's results survive when the objective is a training loss rather than a function, and
which do not. The answer to section 11's question, why the update looks the way it does, is the
subject of that lesson.
