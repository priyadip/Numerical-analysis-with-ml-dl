# Audit: Part 12, Optimization

A part-level audit. [FINAL_AUDIT.md](FINAL_AUDIT.md) carries the current overall position and
links to every part-level audit, so that list is maintained in one place rather than repeated
here and going stale each time a part lands.

Everything below is measured from the repository by [`run_all.py`](run_all.py). No number in
this document was typed in by hand.

---

## Verdict

**Part 12 passes.** All seven lessons exist, all are executed, tested, numerically verified and
fully solved, with no placeholders, no unpaired files and no hardcoded dimensions.

| Check | Result |
|---|---|
| Lessons written | **7 of 7** |
| Notebooks executed end to end | **90 of 90 pass** (every notebook in the repository) |
| Automated tests | **9180 of 9180 pass** (296 new) |
| Assertions inside the Part 12 lessons | **104, all passing** |
| Concepts covered from the sources | **22 new, 485 of 514 total** |
| Concepts outstanding in Part 12 | **0** |
| Worked exercise solutions | **147**, one for every exercise |
| Generality scan | **0 candidates across the repository** |
| Mangled escape scan | **0 control characters across the repository** |

---

## A. Curriculum completeness

Part 12 was planned as 7 lessons. **All seven exist.**

| Lesson | Title | Figures | Assertions | Words | Exercises |
|---|---|---:|---:|---:|---:|
| 84 | Optimization Fundamentals | 1 | 21 | 2485 | 21 |
| 85 | Derivative Free Optimization | 1 | 14 | 2393 | 21 |
| 86 | Gradient and Newton Methods | 1 | 15 | 2877 | 21 |
| 87 | Quasi-Newton and Trust Region Methods | 1 | 15 | 2958 | 21 |
| 88 | Constrained Optimization | 1 | 13 | 2504 | 21 |
| 89 | Stochastic Optimization | 1 | 13 | 2558 | 21 |
| 90 | Proximal and Composite Optimization | 1 | 13 | 2348 | 21 |
| **total** | | **7** | **104** | **18123** | **147** |

**Nothing planned was dropped.**

Seven new library modules back the part:

| Module | Lines | Public functions | Test lines | Tests |
|---|---:|---:|---:|---:|
| `optimize` | 787 | 21 | 293 | 49 |
| `derivfree` | 540 | 11 | 245 | 37 |
| `gradient` | 705 | 14 | 262 | 41 |
| `quasinewton` | 734 | 15 | 320 | 45 |
| `constrained` | 677 | 17 | 301 | 48 |
| `stochastic` | 637 | 12 | 257 | 40 |
| `proximal` | 570 | 17 | 250 | 36 |
| **total** | **4650** | **107** | **1928** | **296** |

The solutions are [`part12_optimization.md`](../solutions/part12_optimization.md):
8719 lines, 147 solutions, 120 executable code blocks, all of which run.

## B. Source completeness

Part 12 covers **22 of 22** concepts attributed to it, so the part is
marked complete in [`coverage_report.md`](coverage_report.md).

| Source | Chapter | Covered in Part 12 | Covered repository wide |
|---|---|---|---|
| Sauer | 13 Optimization | 7 of 7 | 7 of 7 |
| Sauer | Appendix A | 1 of 1 | 3 of 3 |
| Supplementary | 14 topics no source covers | 14 of 14 | 14 of 14 |

The supplementary concepts are:

- Optimality conditions, convexity and the Hessian test, lesson 84
- Conditioning of an optimization problem, lesson 84
- Wolfe conditions and backtracking line search, lesson 86
- Secant condition, BFGS and L-BFGS, lesson 87
- Trust region methods, lesson 87
- Lagrange multipliers and the KKT conditions, lesson 88
- Penalty, barrier and projected gradient methods, lesson 88
- Stochastic and minibatch gradient descent, and gradient noise, lesson 89
- Momentum, Nesterov acceleration and the heavy ball method, lesson 89
- Adaptive step sizes: AdaGrad, RMSProp, Adam and AdamW, lesson 89
- Learning rate schedules and the convergence rate of SGD, lesson 89
- The proximal operator and proximal gradient descent, lesson 90
- ISTA and FISTA acceleration for composite problems, lesson 90
- L1 regularization, LASSO and the soft thresholding operator, lesson 90

None of these appears in a supplied source. The books treat optimization thinly or not at all,
which is expected: it is a field in its own right and the standard numerical analysis texts stop
at Newton's method for a system. A graduate treatment without the material above would be
incomplete, and Change 005 in
[`CURRICULUM_CHANGE_LOG.md`](../_planning/CURRICULUM_CHANGE_LOG.md) records why lessons 89 and 90
were added to carry the mathematical treatment that Part 13 would otherwise have had to improvise.

## C. What the measurements changed

Every claim in the seven lessons was checked against a measurement, and where the two
disagreed the question asked was which of them was wrong. Below are the 19 cases
where the answer turned out to be the code or the claim rather than the test, in 4
groups: claims the measurement did not support, sign and convention errors, diagnostics that
could stop a solve, and exercises whose expected answer did not survive being run.

### Claims the measurement did not support

**`adaptive_loses_when_well_scaled` was false: AdaGrad won at a scaling of 1.** The lesson's
intended finding was that adaptive methods pay for their rescaling on a well scaled problem.
Measured, AdaGrad was ahead there too. The measurement was restructured to run the plain methods at
both a blind step of $0.01$ and at the best $1/L$ step, and the finding became two honest ones
instead of one wrong one: **everything that diverged was non-adaptive**, and the adaptive margin
changes from $2.5$ to $54$ between the two scalings rather than changing sign.

**`recovery_improves_with_samples` was false: the fraction fell at 400 samples.** The penalty was a
fixed multiple of $\lambda_{\max}$, and $\lambda_{\max}$ itself changes with $N$, so a fixed multiple
is not a fixed penalty. Replaced with `false_positives_fall_with_samples` and
`the_rule_separates_them`, plus a note saying explicitly that the normalization is wrong as $N$
changes. Solution 90.4.3 completes the story: with $\lambda \sim \sigma\sqrt{\log m/N}$ the
$k\log(m/k)$ law fits with a constant of $2.13$ and a worst miss of $51\%$; with a fixed multiple of
$\lambda_{\max}$ the constant ranges over a factor of five and half the grid never recovers.

**BFGS's `ratios_fall_towards_zero` claimed a limit a finite run cannot establish.** Superlinear
convergence means a ratio tends to zero, which is a statement about infinitely many terms. The run
reaches the rounding floor in six steps. Replaced with `ratios_are_far_below_one` and
`whether_it_tends_to_zero_is_not_resolvable`, and the measurement now asserts the **negative**
statement it can support, that the second ratio grows by a factor of $3.2\times10^{7}$ so the
convergence is not quadratic.

**`every_run_converged_somewhere` was false for Lennard-Jones: one run hit the step cap.** Split into
`almost_every_run_converged` and `matches_the_published_values`, the second checking the global
minima against the literature exactly.

**`every_wrong_stop_is_a_genuine_minimum` was false: the four variable stop is a saddle.** Split into
`wrong_stops_that_are_minima`, `wrong_stops_that_are_saddles` and `it_stops_at_saddles_too`. Solution
86.4.2 then measured the proportions over 100 random starts per dimension and found the emphasis
backwards: saddles are $1$ to $2$ per cent of outcomes and **wrong local minima are $7$ to $21$ per
cent**. The saddle is detectable with one Hessian; the wrong minimum is not detectable at all by a
local method.

**Solution 89.2.4's derivation was applied where its hypothesis fails.** AdaGrad's effective step
falls like $1/\sqrt k$ when the gradients keep a constant size. In an ordinary run they shrink, and
the measured decay is $k^{-0.13}$ rather than $k^{-0.5}$. The solution now runs both cases, holding
the iterate still to isolate the hypothesis, and states the conclusion with the condition attached.

**Solution 87.5.3 could not construct what the exercise asked for on the obvious problem.** Over 20
random starts on Rosenbrock a longer L-BFGS memory is better at every size and $m = 2$ beats $m = 3$
on only $2$ to $4$ starts. A construction that does work needs the curvature to vary by orders of
magnitude along the path; on such a problem $m = 40$ is $52$ per cent slower than $m = 2$, and the
mechanism is visible in the stored $\rho = 1/(s^{T}y)$, which spans eight orders of magnitude over
fourteen steps.

### Sign and convention errors

**`constrained.py` mixed two Lagrangian sign conventions and the KKT residuals were 2.0 and 0.8.**
`kkt_residual` used $L = f + \lambda^{T}c$ and `augmented_lagrangian` used $L = f - \lambda^{T}c$, so
the residual at the stated exact answers was of order one. Unified on $c \le 0$, $L = f +
\lambda^{T}c$, stationarity $\nabla f + J^{T}\lambda = 0$. The equality problem's multiplier became
$-2/n$, the augmented update became $\lambda \leftarrow \lambda + \mu c$, and the residual at the
stated answers fell to $6.7\times10^{-16}$.

**The library's `augmented_lagrangian` is an equality method and gives a wrong answer on
inequalities, silently.** Its penalty $\tfrac{\mu}{2}\lVert c\rVert^{2}$ punishes $c_i < 0$ as hard
as $c_i > 0$, so on the triangle problem it converges to a point $0.24$ and $0.75$ away from the
answer with no sign of trouble. Solution 88.3.1 records this and implements Rockafellar's form, whose
penalty acts through $\max(0, \lambda + \mu c)$, which reaches $3\times10^{-10}$ and the exact
multipliers.

**Solution 88.3.5 found that projecting a Newton step is simply wrong.** The optimality condition
$x = P_C(x - \alpha\nabla f)$ is about a Euclidean projection of a **gradient** step. Substituting a
Newton step mixes two metrics, and the fixed point is not the KKT system: measured, projected Newton
converges, stays feasible, decreases the objective, and stops $0.39$ to $1.05$ from the exact answer.
Projecting in the Hessian metric, which is exactly lesson 87's trust region subproblem, is exact in
**one** step.

### Diagnostics that could stop a solve

**A BFGS secant diagnostic raised `LinAlgError` and killed a run.** `np.linalg.solve(inverse, move)`
was called inside a measurement in lesson 88 while pushing the penalty weight to $10^{16}$, where the
matrix is singular by design. Wrapped in try/except with `float("inf")` appended: a diagnostic must
not be able to stop a solve.

**Divergence had to become a result rather than a warning.** `stochastic_descent` is deliberately run
past its stability edge in several measurements, so the whole driver is wrapped in `np.errstate`, and
every measurement that sweeps a step treats a non-finite result as data. Solution 89.4.2 uses that
to bisect the stability edge for three methods over four decades of $\kappa$.

**Solution 90.3.2 found a backtracking test that ratchets its own step to zero.** Once the objective
is at its rounding floor, both sides of the descent inequality are noise, the test fails about half
the time, and the step halves each time: it falls from $1.0$ to $10^{-9}$ within a few hundred
iterations and the method silently stops. A slack of a few $\varepsilon\lvert f\rvert$ fixes it. The
reported step count is unaffected, which is why it is easy to miss.

### Solutions where the exercise's expectation did not survive

**85.3.4, the restart rule.** The lesson suggested restarting Nelder-Mead with the same simplex
around the current best point. Measured, that repairs nothing: the eight variable failure stops at
$1.994$ either way and costs $25$ to $40$ per cent more evaluations. A **randomized** restart repairs
it in four of five trials at $2.6$ times the cost. The advice was narrowed to "a restart helps only
if it changes the geometry".

**85.3.3, Fibonacci search.** The first implementation lost by five per cent to golden section. The
classical final step, a nudged comparison at the meeting point, had been omitted, and omitting it
costs **exactly a factor of two** in the final bracket. With it the bracket is exactly $(b-a)/F_n$
and the method wins by $17.1$ per cent.

**86.4.3, the Armijo constant.** The exercise asks for $c_1$ against the iteration and evaluation
counts over five decades. Over six decades below $0.1$ the counts are **bit identical**, because
backtracking by halving from $\alpha = 1$ lands on the same power of two. At $c_1 = 0.5$ the
quadratic run goes from never converging in $200000$ steps to converging in $2139$.

**87.4.3, the trust region rejection rate.** The exercise asks to relate it to lesson 86's fraction of
indefinite Hessians. They are not related: the indefinite fraction rises monotonically to $1.00$ at
forty variables, where the rejection rate is the **lowest** of the six sizes. A trust region steers
toward regions where its model is good, so a statistic about a box does not predict what a method
meets.

**90.4.1, the FISTA speedup against conditioning.** On five of six problems FISTA is **slower** than
ISTA, by seven to fourteen per cent, including on the problem with a restricted condition number of
$6\times10^{9}$. The variable that does predict the speedup is how long the support takes to settle,
and solution 90.4.2 measures the two together: the speedup runs $1.05, 1.13, 1.14, 1.31, 1.71, 2.51,
2.96$ monotonically as the settling time grows from $11$ to $998$ steps.

## D. Negative results, reported as such

Ten results in this part are negative, and each is marked in the module or the solution rather
than replaced with a problem that would have worked.

| Where | What was asked | What was measured |
|---|---|---|
| `derivfree` | whether a same-simplex restart repairs Nelder-Mead | it does not; only a randomized restart does |
| `derivfree` | the parabolic order in double precision | not measurable; five usable points give 1.21 against 1.325 |
| `quasinewton` | whether BFGS is superlinear | not resolvable in double precision; only "not quadratic" is provable |
| `quasinewton` | whether a longer L-BFGS memory ever hurts | not on Rosenbrock over 20 starts; only on a constructed shelf |
| `gradient` | whether Jacobi preconditioning helps | it does not, and at $\kappa = 10^{4}$ it is worse |
| `gradient` | whether $c_1$ is a useful dial | no effect over six decades, then a factor of 93 |
| `constrained` | whether a threshold exists for the augmented Lagrangian | not on a convex problem; the rate is below 1 for every $\mu > 0$ |
| `stochastic` | whether momentum's $\sqrt\kappa$ survives noise | it does not; the gain is $0.80$ to $0.99$ up to $\kappa = 956$ |
| `stochastic` | whether the rounding floor ever binds | never; the crossover batch is $4\times10^{11}$ data sets |
| `proximal` | whether FISTA beats ISTA on these instances | not on five of six; both converge linearly, so neither quoted rate applies |

## E. Where the sources are wrong or incomplete

Four places where the measurement contradicts a statement that appears in standard treatments.

**"Kantorovich's bound is attained."** It is attained from four variables up. At two it is reached
only to $0.71$ of its value, and solution 86.1.2 derives why: from an equal energy start the two
dimensional iteration collapses onto one eigendirection after a single step, with an asymptotic rate
of $1/\sqrt2$ regardless of $\kappa$. Measured at $\kappa = 10^{3}$ it is $0.707140$ against
$1/\sqrt2 = 0.707107$. **A first order method must not be tested in two dimensions.**

**"BFGS is superlinearly convergent."** True and unmeasurable. Solution 87.5.1 shows the
Dennis-Moré condition asks the approximation to be exact only along the directions taken, which is
what BFGS supplies and no more, and the measurement can establish the negative statement (not
quadratic, second ratio growing by $3.2\times10^{7}$) but not the positive one.

**"Gradient clipping fixes a step that is too large."** It removes divergence and does not restore
convergence. Solution 89.3.2 measures the largest stable step growing exactly in inverse proportion
to the clip level, to $1.5\times10^{6}$ times $2/L$, while the final distance grows steadily with the
step: clipping converts divergence into a large noise ball.

**"The modified Cholesky of Gill, Murray and Wright is the efficient repair."** In flop count it wins
by more than twenty to one against the doubling loop. In wall clock at $n = 120$ it loses by a factor
of thirty, because the doubling loop's attempts are LAPACK calls. And its shift is six hundred to
nine hundred times larger than the minimum the doubling loop finds, so the model it produces is
heavily damped. Solution 86.3.1 reports all three.

## F. Trefethen and Bau

**Nothing in Part 12 is attributed to Trefethen and Bau.** The book covers numerical linear algebra
and has no optimization material, and the concept map assigns none of Part 12's concepts to it.

The debt is nonetheless real and is cited where it applies. Lesson 86's exercise 5.1 explains why
steepest descent has $\kappa$ where conjugate gradients has $\sqrt\kappa$ in terms of the polynomial
each method implicitly builds, which is Trefethen and Bau's Krylov material from Part 4 applied to
optimization; the same $\sqrt\kappa$ reappears in lesson 89's heavy ball analysis for the same
Chebyshev reason. Lesson 87's trust region subproblem is solved through the eigendecomposition of
Part 6. Lesson 90's nuclear norm prox is the singular value decomposition of lesson 41 with soft
thresholding applied to the spectrum, and solution 90.3.4 measures the rank dropping exactly, which
is the $\ell_1$ story one level up.

## G. What is not here

Four things a full treatment would include and this part does not.

**Linear programming has no lesson.** The simplex method, duality theory and the interior point
treatment of linear programs are a subject of their own. Solution 88.3.2 and 88.3.3 build an active
set method and a primal-dual interior point method for a quadratic program, which is where the two
ideas meet, and 88.4.3 measures the crossover: the active set method needs $2$ to $108$ iterations
growing with the problem, and the primal-dual method needs $13$ to $18$ regardless.

**Semidefinite programming is absent.** It needs matrix variables and a cone constraint, and the only
place it appears here is the nuclear norm of solution 90.3.4, which is the one case with a closed
form prox.

**Derivative free trust region methods appear only as a remark.** Solution 85.3.5 measures pattern
search against Nelder-Mead and finds the reliability costs about a hundredfold, then notes that
model based methods narrow the gap. Building one needs interpolation set management, which is a
different subject.

**Automatic differentiation is used once and not built.** Solution 84.3.3 implements forward mode
dual numbers to get an exact derivative, and reverse mode, which is what makes lesson 95's
backpropagation possible, is left to Part 13 where it belongs.

## H. Reproducibility

Every lesson fixes `rng = np.random.default_rng(42)` in the standard preamble, and every function in
the seven new modules that uses randomness takes a `seed` argument defaulting to 42. Re-running
`verification/run_all.py` reproduces every number in this document.
