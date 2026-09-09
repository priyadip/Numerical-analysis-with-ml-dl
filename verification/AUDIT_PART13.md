# Audit: Part 13, Stochastic and Monte Carlo Methods

A part-level audit. [FINAL_AUDIT.md](FINAL_AUDIT.md) carries the current overall position and
links to every part-level audit, so that list is maintained in one place rather than repeated
here and going stale each time a part lands.

Everything below is measured from the repository by [`run_all.py`](run_all.py). No number in
this document was typed in by hand.

---

## Verdict

**Part 13 passes.** All three lessons exist, all are executed, tested, numerically verified and
fully solved, with no placeholders, no unpaired files and no hardcoded dimensions.

| Check | Result |
|---|---|
| Lessons written | **3 of 3** |
| Notebooks executed end to end | **93 of 93 pass** (every notebook in the repository) |
| Automated tests | **9350 of 9350 pass** (170 new) |
| Assertions inside the Part 13 lessons | **48, all passing** |
| Concepts covered from the sources | **12 new, 497 of 514 total** |
| Concepts outstanding in Part 13 | **0** |
| Worked exercise solutions | **63**, one for every exercise |
| Generality scan | **0 candidates across the repository** |
| Mangled escape scan | **0 control characters across the repository** |

---

## A. Curriculum completeness

Part 13 was planned as 3 lessons. **All three exist.**

| Lesson | Title | Figures | Assertions | Words | Exercises |
|---|---|---:|---:|---:|---:|
| 91 | Random Number Generation | 1 | 14 | 2010 | 21 |
| 92 | Monte Carlo Methods | 1 | 16 | 2203 | 21 |
| 93 | Brownian Motion and Stochastic Differential Equations | 1 | 18 | 2069 | 21 |
| **total** | | **3** | **48** | **6282** | **63** |

**Nothing planned was dropped.**

Three new library modules back the part:

| Module | Lines | Public functions | Test lines | Tests |
|---|---:|---:|---:|---:|
| `rng` | 532 | 19 | 291 | 49 |
| `montecarlo` | 604 | 17 | 275 | 60 |
| `sde` | 565 | 17 | 280 | 61 |
| **total** | **1701** | **53** | **846** | **170** |

The solutions are [`part13_stochastic_methods.md`](../solutions/part13_stochastic_methods.md):
3393 lines, 63 solutions, 45 executable code blocks, all of which run.

## B. Source completeness

Part 13 covers **12 of 12** concepts attributed to it, so the part is
marked complete in [`coverage_report.md`](coverage_report.md).

| Source | Chapter | Covered in Part 13 | Covered repository wide |
|---|---|---|---|
| Sauer | 9 Random Numbers | 10 of 10 | 10 of 10 |
| Supplementary | 2 topics no source covers | 2 of 2 | 2 of 2 |

The supplementary concepts are:

- Mersenne Twister and PCG64 generators, statistical tests, lesson 91
- Variance reduction: antithetic variates and control variates, lesson 92

Neither appears in a supplied source. Sauer's chapter 9 covers linear congruential generators and
the normal transforms but stops before the modern generators and before any statistical test, and it
mentions variance reduction only in passing. A graduate treatment without them would leave the
reader unable to judge a generator or to make a Monte Carlo estimate converge faster than
``1/sqrt(N)``, which are the two practical questions this part exists to answer.

## C. What the measurements changed

Every claim in the three lessons was checked against a measurement, and where the two
disagreed the question asked was which of them was wrong. Below are the 19 cases
where the answer turned out to be the code or the claim rather than the test, in 4
groups: results the measurement contradicted, quantities that were the wrong quantity, fits taken
where the signal was not, and exercises whose expected answer did not survive being run.

### Results the measurement contradicted

**The RANDU plane count came out at 26086 rather than 15.** The first implementation asserted a
plane normal of $(a^2 \bmod m,\, -2a,\, 1)$, which is a valid relation but not a short one, so the
combination spanned tens of thousands of multiples of the modulus. Replaced with a bounded search for
the **shortest** integer relation, which finds $(9, -6, 1)$ without being told it, gives exactly
$15$ planes, and finds nothing at all for the other four generators. That last part is what makes the
measurement a measurement: a search that always returned $15$ would pass the first check and fail
this one.

**The polar method's acceptance rate came out at exactly $0.5$**, which is not $\pi/4$ and is too
round a number to be a coincidence. The block size was being counted in output values while the
acceptance was counted in candidate pairs, so the two differed by the factor of two that each
accepted pair produces. After the fix the measured acceptance is $0.785352$ against
$\pi/4 = 0.785398$, five digits.

**The antithetic variance ratio was predicted at twice its true value.** The formula
$2/(1+\rho)$ compares the variance of a **pair average** against the variance of a single
evaluation, which is not a comparison anyone would make: the paired estimator spends two evaluations
per pair. At equal evaluations the ratio is $1/(1+\rho)$. The measured values then match to under one
per cent, and the correction changes the symmetric case from "no help" to a **loss of a factor of
two**, which is a different piece of advice.

**The `oscillating` integrand was labelled symmetric and is monotone.** $\cos(\pi x/2)$ decreases on
$[0,1]$, so $f(1-u) = \sin(\pi u/2)$ and the pair correlation is $-0.47$, not $+1$. A genuinely
symmetric integrand, $\prod(1 + \cos 2\pi x_i)$, was added, and it gives $\rho = +1.000000$ exactly.

**The ziggurat's layer areas exceeded the area under the density by $1.2$ per cent**, which looked
like a failed table solve. Two things were wrong and one was not. The bisection was stopping at the
smallest right edge for which the stack fits rather than driving the closure residual to zero, which
was a real bug and is now solved to $10^{-16}$. The labels for the innermost and outermost layers
were swapped. And the excess itself is **correct**: the rectangles cover the curve, so their total
must be larger, and the gap is the rejection rate. It is $0.81$, $0.44$ and $0.28$ per cent at $32$,
$128$ and $256$ layers, which is a more useful number than the one the check was looking for.

**Solution 92.3.5's predicted variance for the arctangent estimator was wrong by a factor of
$1.57$.** The integral $\int_0^1(1+x^2)^{-2}dx$ is $\tfrac14 + \tfrac\pi8$, and it had been evaluated
as though the two terms were subtracted. The corrected prediction matches the measurement, and the
ranking of the three estimators changes from $1 : 3.03 : 10.25$ to $1 : 1.93 : 6.52$.

### Quantities that were the wrong quantity

**The Monte Carlo exponent was fitted from medians of absolute errors and came out at $-0.40$ to
$-0.58$.** The mean square error is exactly $\sigma^2/N$, so its square root estimates the quantity
being fitted with a relative spread of $1/\sqrt{2R}$; a median of a dozen absolute errors estimates
something close to it with several times the noise, and at twelve repeats that noise is larger than
the effect. Switched to a root mean square over sixty repeats, after which every exponent is within
$0.017$ of $-0.5$ and the spread across dimensions is $0.0166$.

**The Halton comparison used a median over nine seeds for its random baseline**, which put the
random exponent at $-0.879$ in two dimensions rather than near $-0.5$ and made the advantage look
like $3$ when it is $19$. Switched to the same root mean square estimator.

**The weak order was measured against the true mean and came out at $0.18$.** The residual sampling
error, common to every step size, was far larger than the bias being measured. Measuring against the
**same sample's** exact mean cancels it, which is lesson 92's control variate applied to the
measurement rather than to the integral, and the errors then fall monotonically.

**The antithetic gain on the option price was reported as a number rather than predicted.** Replaced
with the pair correlation and lesson 92's formula: $\rho = -0.5009$ gives a predicted error ratio of
$1.4155$ against a measured $1.4978$, and it explains **why** the gain is modest, which the bare
number did not.

### Fits taken where the signal was not

**The strong and weak orders cannot be fitted over the same step range, and doing so measures
noise.** The strong error is of order $\sqrt h$ and needs fine steps to be inside its regime; the
weak error is a difference of two means of order $h$ and at fine steps drops below the residual
sampling noise. Fitted over $h$ from $1/16$ to $1/256$ and from $1/2$ to $1/32$ respectively, the
orders are $0.4974$ and $0.9950$. Fitted over a single range, one of them is wrong by a factor of
three. The measurement now states its ranges and the reason.

**The Monte Carlo crossover was measured on a separable integrand and found no crossover.** That is
correct and it is not what the exercise was asking. $\prod e^{x_i}$ factors, so a product rule
inherits its one dimensional accuracy at every dimension and never loses. A discontinuous integrand
was added, on which the crossover is at dimension $4$, and both are now reported: the honest
conclusion is that **the crossover depends on the integrand's smoothness and separability, not on
the dimension alone**, and the usual argument naming Simpson and deriving $d = 8$ has the wrong
variable in it.

### Solutions where the exercise's expectation did not survive

**91.3.4, the gap test and birthday spacings.** The exercise asks for a generator each test catches
that the others do not. Measured, **neither test catches anything**: the gap ratios run $0.54$ to
$1.17$ with RANDU at $0.997$, the best value in the table, and every birthday ratio including the
modern generator's sits at $0.93$ to $0.95$. Reported as the negative result it is, with the
observation that the uniform $0.94$ is a defect of the Poisson approximation at these parameters
rather than of any generator.

**91.4.2, the samples needed to detect RANDU.** At six cells per axis the failure is **invisible at
two million samples**, while four, eight, ten, fourteen and twenty all catch it at fifty thousand.
With $216$ cells and $15$ planes the spacing happens to distribute the points evenly. A grid that
aligns with the structure hides it, so a single cell count is not a test.

**91.4.3, the cost of a normal.** The usual claim is that the polar method is faster than Box-Muller
because it avoids trigonometry. Measured, it is **ten to fifteen per cent slower**, because the
trigonometric calls are vectorized and the rejection branch is not. The ordering is a property of the
machine, so the solution reports ratios rather than nanoseconds.

**92.4.3, whether the savings multiply.** Separately the antithetic pairing gains $4.19$ and the
control variate $333.5$, whose product is $1398$. Together they give $489$, about a third. They
exploit the same structure, so once the control has removed the monotone trend the pairing has little
left to work with.

**93.3.4, the barrier option.** The expectation was that the path dependent payoff would separate
Euler-Maruyama from Milstein. It does not: they agree to $10^{-4}$. What the measurement finds
instead is that the **barrier price moves twenty per cent between $16$ and $2048$ steps**, where the
European moves two, because the discrete maximum over grid points misses crossings between them. The
dominant error is in the payoff's discretization rather than the path's, and it falls only like
$O(\sqrt h)$.

**93.4.3, the weak error against payoff smoothness.** The theorem asks for a smooth $g$, so the
expectation was that a discontinuous payoff would lose the order. It does not: the measured weak
orders are $1.013$, $0.981$, $1.124$ and $1.005$ for the identity, a square, a call payoff and an
indicator. The hypothesis is sufficient and not necessary, because the law of $X(T)$ has a smooth
density that integrates the roughness away. **What changes is the constant**, by a factor of $11$.

## D. Negative results, reported as such

Eight results in this part are negative, and each is marked in the module or the solution rather
than replaced with a problem that would have worked.

| Where | What was asked | What was measured |
|---|---|---|
| `rng` | whether uniformity or correlation catches RANDU | neither; RANDU's lag one correlation is smaller than PCG64's |
| `rng` | a generator the gap test catches | none; RANDU scores best of six on it |
| `rng` | a generator birthday spacings catches | none; every ratio including PCG64's is 0.94 |
| `rng` | whether the polar method is faster than Box-Muller | it is 10 to 15 per cent slower on this machine |
| `montecarlo` | the crossover dimension against a grid | none on a separable integrand, at any dimension |
| `montecarlo` | whether a better quadrature rule moves the crossover | Gauss moves it not at all; only smoothness does |
| `montecarlo` | whether variance reduction savings multiply | they do not; 4.19 and 333.5 give 489, not 1398 |
| `sde` | whether Milstein helps a path dependent payoff | it does not; the payoff's own discretization dominates |

## E. Where the sources are wrong or incomplete

Four places where the measurement contradicts a statement that appears in standard treatments.

**"Monte Carlo beats a grid above about eight dimensions."** The argument sets $N^{-k/d}$ against
$N^{-1/2}$ for a $k$-th order rule. It assumes the integrand is generic. On a separable one the grid
never loses at any dimension tested, and on a discontinuous one sampling wins at **four**, because
the grid's rate there is $N^{-1/d}$ and no order of rule improves it. Solution 92.4.2 repeats the
whole measurement with Gauss-Legendre and finds the crossover unmoved.

**"Antithetic variates help a little or not at all."** They cost a factor of two on a symmetric
integrand, measured at $0.504$ against a predicted $0.5$, because the pairing then wastes exactly
half the budget.

**"Milstein is better than Euler-Maruyama."** It is better in the strong sense by a factor of $31$
and **not at all** in the weak one, at a measured weak gain of $0.970$. Since most SDE calculations
want an expectation, the extra term is usually wasted work, and its one weak use is inside a
multilevel estimator where the strong order sets the cost.

**"Weak order one needs a smooth payoff."** The theorem's hypothesis is sufficient and not necessary.
A discontinuous indicator gives a measured weak order of $1.005$, because the density of the solution
is smooth and integrates the roughness away. The constant grows by a factor of $11$, and that is the
whole of the practical difference.

## F. Trefethen and Bau

**Nothing in Part 13 is attributed to Trefethen and Bau.** The book covers numerical linear algebra
and has no material on random number generation, Monte Carlo or stochastic differential equations,
and the concept map assigns none of Part 13's concepts to it.

One debt is cited where it applies. Solution 91.3.1 computes the spectral test by Lenstra-Lenstra-Lovasz
lattice reduction, whose inner loop is the Gram-Schmidt orthogonalization of Part 5 applied over the
integers, and whose numerical difficulty, that classical Gram-Schmidt loses orthogonality, is the
subject of lesson 30. The reduction here works in exact enough arithmetic for the sizes used, and the
same computation at cryptographic sizes needs the modified Gram-Schmidt or the Householder approach
of lesson 31 for exactly the reason that lesson gives.

## G. What is not here

Four things a full treatment would include and this part does not.

**Markov chain Monte Carlo is absent.** Metropolis-Hastings, Gibbs sampling and Hamiltonian Monte
Carlo are the tools for sampling a distribution known only up to a constant, which is a different
problem from the ones here: the samples are correlated, the error analysis needs an integrated
autocorrelation time rather than $\sqrt N$, and convergence has to be diagnosed rather than bounded.
It is a subject of its own and it belongs with Bayesian computation.

**Rare event simulation appears only as importance sampling.** Solution 92.3.1 measures a gain of
$8\times10^{14}$ when the proposal matches the integrand, which is the mechanism, but splitting,
cross entropy methods and the theory of efficient estimators for probabilities of order $10^{-9}$ are
not here.

**Stochastic partial differential equations are absent.** They need Part 11's spatial discretization
and this part's temporal one together, plus a treatment of noise that is white in time and coloured
in space, and the interaction is not a combination of the two parts as written.

**Jump processes are absent.** Every equation here is driven by Brownian motion alone. Adding a
Poisson jump term changes the Ito calculus, the order conditions and the simulation, and the standard
schemes for it are a separate family.

## H. Reproducibility

Every lesson fixes `rng = np.random.default_rng(42)` in the standard preamble, and every function in
the three new modules that uses randomness takes a `seed` argument defaulting to 42. The one place
this part depends on timing rather than arithmetic, solution 91.4.3's cost per normal, reports ratios
rather than absolute times and says so. Re-running `verification/run_all.py` reproduces every other
number in this document.
