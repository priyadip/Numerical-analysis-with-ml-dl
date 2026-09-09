# 92. Monte Carlo Methods

**Part 13: Stochastic and Monte Carlo Methods**

## Learning objectives

By the end of this lesson you will be able to:

1. Estimate an integral by sampling, and attach an error bar you can defend.
2. Say why the error falls like $1/\sqrt N$ and why the dimension does not appear.
3. Say when a quadrature rule beats sampling and when it cannot, and why the usual answer is
   incomplete.
4. Apply antithetic variates, control variates and stratification, and predict what each will save
   before running it.
5. Use a low discrepancy sequence, and say where its advantage ends.

## Prerequisites

Lesson 91 (the samples come from somewhere, and in three dimensions it matters where). Lesson 63
(Simpson's rule, the competitor here). Lesson 66 (multidimensional quadrature, where the curse of
dimensionality was first stated).

---

## 1. Sampling as a quadrature rule

An integral over the unit cube is an average:

$$
I = \int_{[0,1]^d} f(x)\,dx = \mathbb{E}\big[f(U)\big] , \qquad U \text{ uniform on the cube.}
$$

So draw $N$ points and take the mean. The estimator is unbiased for any $N$, and by the central
limit theorem its error is approximately normal with standard deviation $\sigma/\sqrt N$, where
$\sigma$ is the standard deviation of $f(U)$.

That gives an error bar for free, computed from the same sample. The bar is only worth having if it
is honest, so measure it: run the estimator four hundred times and compare the bar it reports with
the spread it actually has.

```python
from nalib import montecarlo as mc

out = mc.the_error_bar_is_honest()
print(f"{out['repeats']} independent runs of {out['samples']} samples in "
      f"{out['dimension']} dimensions")
print(f"  spread of the estimates:   {out['actual_spread']:.6f}")
print(f"  average reported bar:      {out['claimed_spread']:.6f}")
print(f"  ratio:                     {out['ratio']:.4f}")
print(f"  runs landing within one bar: {out['fraction_within_one_bar']:.3f}")
print(f"  a normal predicts:           {out['predicted_fraction']:.3f}")

assert out["the_bar_is_honest"]
assert out["coverage_matches"]
```

The reported bar is within two per cent of the true spread, and $69$ per cent of runs land inside it
against the $68.3$ per cent a normal predicts. **The bar is honest**, which is the first thing to
establish about any estimator that reports one.

## 2. The rate, and what is missing from it

The error is $\sigma/\sqrt N$, so it falls like $N^{-1/2}$. Four times the work buys two times the
accuracy, forever. Compared with Simpson's $N^{-4}$ in one dimension that is dreadful.

What is missing from the expression is the dimension. It is not there, and it is not hiding in
$\sigma$ either, which is a property of $f$ rather than of the grid.

```python
from nalib import montecarlo as mc

out = mc.the_error_falls_like_one_over_root_n()
print(f"sample counts: {out['counts']}")
print(f"{'dimension':>11}{'fitted exponent':>18}{'off -1/2 by':>14}")
for row in out["rows"]:
    print(f"{row['dimension']:>11}{row['fitted_power']:>18.4f}{row['off_by']:>14.4f}")
print(f"\nspread of the exponent across dimensions: "
      f"{out['spread_across_dimensions']:.4f}")

assert out["every_power_is_minus_a_half"]
assert out["spread_across_dimensions"] < 0.06
```

From one dimension to twenty the exponent moves by $0.017$. **That is the property that makes Monte
Carlo the only option above about ten variables**, and no quadrature rule has it.

Two details about the measurement matter. The error at each sample count is the root mean square
over sixty repeats, because the mean square error is exactly $\sigma^2/N$ and its square root
therefore estimates the right quantity with a spread of $1/\sqrt{2R}$. A median of a dozen absolute
errors estimates something close to it with several times the noise, and at that noise level the
exponent cannot be resolved to better than about $0.1$, which is larger than the effect.

## 3. Against a grid, honestly

The usual argument: a $k$-th order product rule with $N = n^d$ points has error $O(N^{-k/d})$, and
sampling has $O(N^{-1/2})$, so they cross where $k/d = 1/2$. For Simpson, $k = 4$, giving $d = 8$.

That argument assumes the integrand is generic. Run it on two that are not.

```python
from nalib import montecarlo as mc

out = mc.where_monte_carlo_overtakes_a_grid()
print(f"budget about {out['budget']} evaluations")
current = None
for row in out["rows"]:
    if row["integrand"] != current:
        current = row["integrand"]
        print(f"\n  {current}")
        print(f"{'d':>5}{'per axis':>10}{'points':>10}{'grid error':>14}"
              f"{'sampled error':>16}{'sampling wins':>15}")
    if row["too_expensive"]:
        print(f"{row['dimension']:>5}{row['per_axis']:>10}{row['grid_points']:>10}"
              f"{'too expensive':>14}{'':>16}{'':>15}")
    else:
        print(f"{row['dimension']:>5}{row['per_axis']:>10}{row['grid_points']:>10}"
              f"{row['grid_error']:>14.3e}{row['monte_carlo_error']:>16.3e}"
              f"{str(row['monte_carlo_wins']):>15}")

print(f"\ncrossover on the separable integrand: {out['separable_crossover']}")
print(f"crossover on the discontinuous one:  {out['rough_crossover']}")

assert out["the_grid_never_loses_on_the_separable_one"]
assert out["monte_carlo_wins_early_on_the_rough_one"]
```

**On a separable integrand the grid never loses.** $\prod e^{x_i}$ factors, so a product rule
inherits its one dimensional accuracy at every dimension: at eight variables with five points per
axis the grid error is $1.7\times10^{-4}$ against sampling's $1.2\times10^{-3}$. The crossover
argument does not apply, because $O(N^{-4/d})$ was a bound for a generic integrand and this one is
not generic.

**On a discontinuous integrand the grid loses at four dimensions**, not eight. The error there comes
from the cells the boundary cuts, and there are $O(N^{(d-1)/d})$ of them, so the grid converges like
$N^{-1/d}$ and not $N^{-4/d}$. Smoothness was doing all the work in the original argument.

So the honest statement is that **the crossover depends on the integrand and not only on the
dimension**, and the two things that decide it are smoothness and separability.

## 4. Antithetic variates

Pair each point $u$ with $1-u$ and average the pair. The variance of the pair average is
$\tfrac12(1+\rho)$ times the plain variance, with $\rho$ the correlation between $f(u)$ and
$f(1-u)$. At equal **evaluations**, which is the only fair comparison, the ratio is

$$
\frac{\text{plain variance}}{\text{antithetic variance}} = \frac{1}{1+\rho} .
$$

```python
from nalib import montecarlo as mc

out = mc.antithetic_helps_only_a_monotone_integrand()
print(f"{out['samples']} evaluations in {out['dimension']} dimensions")
print(f"{'integrand':>36}{'rho':>12}{'predicted ratio':>18}{'measured':>13}"
      f"{'error':>12}")
for row in out["rows"]:
    predicted = ("infinite" if row["predicted_variance_ratio"] == float("inf")
                 else f"{row['predicted_variance_ratio']:.3f}")
    print(f"{row['integrand']:>36}{row['correlation']:>+12.6f}{predicted:>18}"
          f"{row['measured_variance_ratio']:>13.3f}{row['antithetic_error']:>12.2e}")

assert out["the_prediction_holds"]
assert out["linear_error"] == 0.0
```

Three regimes, and the formula gets all three.

**A linear integrand gives $\rho = -1.000000$ exactly, and the answer is exact.** The error is
$0.00\times10^{0}$: $f(u) + f(1-u)$ is constant, so a single pair integrates a linear function with
no error at all.

**A monotone integrand gives a real saving**, $\rho = -0.760$ and a measured ratio of $4.165$ against
a predicted $4.166$.

**A symmetric integrand costs a factor of two.** $\rho = +1.000000$, because $f(1-u) = f(u)$
identically, so the second evaluation of each pair carries no information and half the budget is
wasted. The measured ratio is $0.504$ against a predicted $0.5$.

That last row is worth keeping. Antithetic sampling is usually described as a technique that helps a
little or not at all; on the wrong integrand it is a technique that halves your sample.

## 5. Control variates

Take a function $g$ whose integral is known and which correlates with $f$. The estimator

$$
f - \beta\,\big(g - \mathbb{E}g\big)
$$

has the same mean for every $\beta$, and its variance is minimized at $\beta = \operatorname{cov}(f,g)/\operatorname{var}(g)$,
where it becomes $(1-\rho^2)$ times the original.

```python
from nalib import montecarlo as mc

out = mc.a_control_variate_achieves_one_minus_rho_squared()
print(f"{out['samples']} samples in {out['dimension']} dimensions, "
      f"plain error {out['plain_error']:.3e}")
print(f"{'control':>40}{'rho':>12}{'beta':>10}{'predicted':>12}{'measured':>12}"
      f"{'error':>12}")
for row in out["rows"]:
    print(f"{row['control']:>40}{row['correlation']:>+12.6f}{row['beta']:>10.4f}"
          f"{row['predicted_variance_ratio']:>12.2f}"
          f"{row['measured_variance_ratio']:>12.2f}{row['error']:>12.2e}")

print(f"\nworst relative miss: {out['worst_relative_miss']:.2e}")

assert out["the_prediction_holds"]
```

The predicted and measured ratios agree to **zero relative error** in all three cases. That is not a
tolerance being met; $1/(1-\rho^2)$ and the ratio of sample variances are the same algebraic
quantity computed two ways.

The practical content is in what the formula does not mention. It does not matter how $g$ is built,
how expensive it is to state, or whether it resembles $f$ in any way a person would notice. **Only
the correlation matters.** A second order Taylor expansion of the integrand correlates at $0.9985$
and saves a factor of $332$; the linear term correlates at $0.9335$ and saves $7.8$. Searching for a
control variate is searching for correlation and nothing else.

## 6. Stratification

Split the cube into cells and sample each one. The variance becomes the average of the within-cell
variances, so the saving is whatever share of the total variance lay **between** cells.

```python
from nalib import montecarlo as mc

out = mc.stratification_pays_where_the_integrand_is_smooth()
print(f"{out['per_axis']} strata per axis in {out['dimension']} dimensions, "
      f"{out['cells']} cells")
print(f"{'integrand':>32}{'samples':>10}{'plain error':>14}{'stratified':>14}"
      f"{'variance ratio':>16}")
for row in out["rows"]:
    print(f"{row['integrand']:>32}{row['samples']:>10}{row['plain_error']:>14.3e}"
          f"{row['stratified_error']:>14.3e}{row['variance_ratio']:>16.2f}")

assert out["it_pays_more_on_the_smooth_one"]
```

On the smooth integrand the variance falls by $226$, close to the $256$ cells, because a nearly
linear function has almost all its variance between cells and almost none inside one. On the
discontinuous integrand it falls by only $11.5$: the cells the boundary crosses keep their variance
whatever the grid does, and there are enough of them to dominate.

Stratification also has a cost the formula hides. It needs $k^d$ cells, so at twenty dimensions and
two strata per axis it needs a million of them before a single sample is drawn. **Latin hypercube
sampling** is the usual repair, stratifying each axis separately rather than the product.

## 7. Quasi-random points

A random sample is uneven by construction: it clumps and it leaves gaps, and both are $O(N^{-1/2})$
in size. A **low discrepancy sequence** gives that up deliberately. It is not random and does not
try to look random; it tries to be evenly spread at every prefix length.

The star discrepancy measures how uneven a point set is, and Koksma's inequality bounds the
integration error by the discrepancy times the variation of the integrand. So the discrepancy is the
thing to measure.

```python
from nalib import montecarlo as mc

out = mc.the_discrepancy_is_the_reason()
print(f"{'points':>9}{'van der Corput':>18}{'random':>14}{'ratio':>10}")
for n, q, r in zip(out["counts"], out["quasi"], out["random"]):
    print(f"{n:>9}{q:>18.3e}{r:>14.3e}{r / q:>10.2f}")
print(f"\nfitted exponents: van der Corput {out['quasi_power']:.4f}, "
      f"random {out['random_power']:.4f}")

assert out["random_is_root_n"]
assert out["quasi_is_much_faster"]
```

The van der Corput exponent is $-1.0000$ and the random one is $-0.4634$. At $16384$ points the
discrepancy differs by a factor of $108$.

Halton's sequence is one van der Corput sequence per axis, each with a different prime base. Its
discrepancy is $O((\log N)^d/N)$, and that $(\log N)^d$ is the whole story of where it stops working.

```python
from nalib import montecarlo as mc

out = mc.halton_beats_sampling_then_stops()
print(f"sample counts: {out['counts']}")
print(f"{'dimension':>11}{'Halton exponent':>18}{'random exponent':>18}"
      f"{'Halton error':>15}{'random error':>15}{'advantage':>12}")
for row in out["rows"]:
    print(f"{row['dimension']:>11}{row['quasi_power']:>18.4f}"
          f"{row['random_power']:>18.4f}{row['quasi_error_at_the_largest']:>15.3e}"
          f"{row['random_error_at_the_largest']:>15.3e}{row['advantage']:>12.2f}")

assert out["quasi_wins_in_two_dimensions"]
assert out["the_advantage_shrinks"]
```

**In two dimensions Halton is nineteen times more accurate**, with an exponent of $-0.94$ against
$-0.52$. By sixteen dimensions the advantage is $0.80$: it is **worse than random sampling**.

The reason is visible in the exponent column. Halton's degrades steadily, $-0.94, -0.92, -0.87,
-0.75$, while the random exponent sits at $-0.5$ throughout. The higher primes need very many points
before their digits have cycled enough to fill the axis evenly, and at the sample sizes anyone uses
the later coordinates are close to correlated.

Scrambling and the Sobol sequence with a good direction table push the usable dimension up
considerably, which is why financial practice uses them at hundreds of dimensions and plain Halton is
a teaching example.

## 8. The picture

```python
from nalib import montecarlo as mc

fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.0))

counts = [250, 1000, 4000, 16000, 64000]
repeats = 40
f, exact, _ = mc.problems(4)["smooth"]
for label, style in (("random", "o-"), ("Halton", "s--")):
    errors = []
    for n in counts:
        if label == "random":
            draws = np.array([(mc.integrate(f, 4, n, seed=s)["estimate"] - exact) / exact
                              for s in range(repeats)])
            errors.append(float(np.sqrt(np.mean(draws ** 2))))
        else:
            errors.append(abs(mc.quasi_integrate(f, 4, n)["estimate"] - exact) / exact)
    left.loglog(counts, errors, style, ms=5, lw=1.5, label=label)
reference = np.asarray(counts, dtype=float)
left.loglog(reference, 0.02 / np.sqrt(reference), ":", color="0.5", lw=1.2,
            label="N to the minus a half")
left.set_xlabel("samples")
left.set_ylabel("relative error")
left.set_title("four dimensions")
left.legend(fontsize=8)

points = 512
right.plot(*np.random.default_rng(42).random((points, 2)).T, ".", ms=3,
           alpha=0.7, label="random")
right.plot(*(mc.halton(points, 2).T + np.array([[1.15], [0.0]])), ".", ms=3,
           alpha=0.7, label="Halton, shifted right")
right.set_xlim(-0.05, 2.25)
right.set_ylim(-0.05, 1.05)
right.set_aspect("equal")
right.set_title(f"{points} points each")
right.legend(fontsize=8, loc="upper center")

for panel in (left, right):
    panel.grid(alpha=0.3, which="both")
fig.tight_layout(); fig.savefig("../figures/92_montecarlo.png", dpi=110); plt.close(fig)
print("saved ../figures/92_montecarlo.png")
```

![Convergence in four dimensions, and the two point sets side by side](../figures/92_montecarlo.png)

The right panel is the reason for the left one. The random points clump and leave holes; the Halton
points do neither. Both fill the square, and only one of them fills it evenly.

## 9. From scratch

The estimator is one line and the error bar is one more. Almost everything in this lesson is a
variation on the same average.

```python
import numpy as np


def my_monte_carlo(f, dimension, samples, seed=0):
    points = np.random.default_rng(seed).random((samples, dimension))
    values = f(points)
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(samples))


def my_antithetic(f, dimension, samples, seed=0):
    points = np.random.default_rng(seed).random((samples // 2, dimension))
    paired = 0.5 * (f(points) + f(1.0 - points))
    return float(np.mean(paired)), float(np.std(paired, ddof=1) / np.sqrt(paired.size))


f, exact, _ = mc.problems(4)["smooth"]
plain, plain_bar = my_monte_carlo(f, 4, 200000)
folded, folded_bar = my_antithetic(f, 4, 200000)
print(f"exact                {exact:.8f}")
print(f"plain                {plain:.8f} +/- {plain_bar:.2e}")
print(f"antithetic           {folded:.8f} +/- {folded_bar:.2e}")
print(f"the bar shrank by a factor of {plain_bar / folded_bar:.3f}, "
      f"and the square of that is {(plain_bar / folded_bar) ** 2:.3f}")

library = mc.integrate(f, 4, 200000)
print(f"\nthe library's plain estimate agrees: "
      f"{abs(library['estimate'] - plain) < 1e-12}")

assert abs(plain - exact) < 5.0 * plain_bar
assert abs(folded - exact) < 5.0 * folded_bar
```

Eight lines reproduce the variance reduction of section 4.

## 10. Exercises

**Level 1, understanding**

1.1 Say why the Monte Carlo error does not depend on the dimension.

1.2 State the standard error of a Monte Carlo estimate and say what it is an estimate of.

1.3 Say when antithetic variates help, when they do nothing, and when they cost.

1.4 Explain what a control variate needs, and what it does not need.

1.5 Say what low discrepancy means and why it is not the same as random.

**Level 2, derivation**

2.1 Derive the standard error $\sigma/\sqrt N$ and say which theorem makes it a confidence interval.

2.2 Derive the antithetic variance ratio $1/(1+\rho)$ at equal evaluations.

2.3 Derive the optimal control coefficient and the resulting $(1-\rho^2)$.

2.4 Derive the stratified variance and show it never exceeds the unstratified one.

2.5 State Koksma's inequality and use it to explain the van der Corput exponent measured in
section 7.

**Level 3, computational**

3.1 Implement importance sampling and use it on an integrand concentrated in a small region, where
plain sampling wastes almost every draw.

3.2 Implement Latin hypercube sampling and compare it against full stratification at ten dimensions,
where full stratification cannot be run.

3.3 Implement a scrambled Halton sequence and measure how far up in dimension it pushes the
advantage of section 7.

3.4 Implement the Sobol sequence with direction numbers and compare against Halton at 20 and 50
dimensions.

3.5 Implement a Monte Carlo estimate of $\pi$ three ways, and rank them by variance before running
them.

**Level 4, experimental**

4.1 Measure how many samples the estimator needs before its error bar has the coverage a normal
predicts, as a function of the skewness of the integrand.

4.2 Measure the crossover dimension against a product Gauss rule rather than Simpson, and say what
changes.

4.3 Measure the combined effect of antithetic variates and a control variate, and say whether the
savings multiply.

**Level 5, advanced**

5.1 **Why $1/\sqrt N$ cannot be improved.** State the information theoretic reason no unbiased
estimator from $N$ independent samples can do better, and say how quasi-random methods get around
it.

5.2 **When the variance is infinite.** Construct an integrand with a finite integral and an infinite
variance, show the error bar becomes meaningless, and describe the two standard repairs.

5.3 **Randomized quasi-Monte Carlo.** Explain why a scrambled low discrepancy sequence recovers an
error bar that plain Halton cannot give, and what it costs.

## 11. Key takeaways

- **The error bar is honest.** Over four hundred runs the reported bar is within $1.6$ per cent of
  the true spread, with $69$ per cent coverage against a predicted $68.3$.

- **The exponent is $-1/2$ and does not move with the dimension**, varying by $0.017$ between one
  and twenty variables.

- **The textbook crossover argument is incomplete.** On a separable integrand a product Simpson rule
  never loses, at any dimension tested; on a discontinuous one sampling wins from dimension **4**.

- **Antithetic sampling has three regimes**, all predicted by $1/(1+\rho)$: exact on a linear
  integrand, a factor of $4.165$ against a predicted $4.166$ on a monotone one, and a **loss** of a
  factor of two on a symmetric one.

- **A control variate achieves $1/(1-\rho^2)$ and nothing else**, matching to zero relative error at
  ratios of $7.8$, $7.8$ and $332$.

- **Stratification pays where the integrand is smooth**, a factor of $226$ with $256$ cells, against
  $11.5$ on a discontinuous integrand.

- **van der Corput's discrepancy falls like $N^{-1.0000}$** against a random sample's $N^{-0.46}$, a
  factor of $108$ at $16384$ points.

- **Halton wins by $19\times$ at two dimensions and loses at sixteen**, with an exponent degrading
  from $-0.94$ to $-0.75$ while the random one stays at $-0.5$.

## Where this goes next

Lesson 93 stops computing numbers and starts computing paths. A stochastic differential equation is
lesson 67's initial value problem with noise added at every step, and almost everything Part 10
established has to be restated: there are now two orders of accuracy rather than one, the chain rule
acquires an extra term, and the step size controls something that was not there before.
