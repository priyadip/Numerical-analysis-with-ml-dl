# Audit: Part 9, Numerical Differentiation and Integration

A part-level audit. [FINAL_AUDIT.md](FINAL_AUDIT.md) carries the current overall position and
links to every part-level audit, so that list is maintained in one place rather than repeated
here and going stale each time a part lands.

Everything below is measured from the repository by [`run_all.py`](run_all.py). No number in
this document was typed in by hand.

---

## Verdict

**Part 9 passes.** All six lessons exist, all are executed, tested, numerically verified and
fully solved, with no placeholders, no unpaired files and no hardcoded dimensions.

| Check | Result |
|---|---|
| Lessons written | **6 of 6** |
| Notebooks executed end to end | **6 of 6 pass** (66 of 66 across the repository) |
| Automated tests | **7656 of 7656 pass** (1151 new, 5 skipped by design) |
| Assertions inside the Part 9 lessons | **37, all passing** |
| Concepts covered from the sources | **37 new, 384 of 507 total** |
| Concepts outstanding in Part 9 | **0** |
| Worked exercise solutions | **115**, one for every exercise |
| Generality scan | **0 candidates across 323 files** |
| Mangled escape scan | **0 control characters across the repository** |

---

## A. Curriculum completeness

Part 9 was planned as 6 lessons. **All six exist.**

| Lesson | Title | Figures | Assertions | Words | Exercises |
|---|---|---:|---:|---:|---:|
| 61 | Numerical Differentiation | 1 | 7 | 4160 | 19 |
| 62 | Newton-Cotes Quadrature | 1 | 8 | 3296 | 19 |
| 63 | Richardson, Romberg and Euler-Maclaurin | 1 | 9 | 3393 | 19 |
| 64 | Adaptive Quadrature | 1 | 3 | 3027 | 18 |
| 65 | Gaussian Quadrature | 1 | 6 | 4165 | 19 |
| 66 | Improper and Multiple Integrals | 1 | 4 | 3792 | 21 |
| **total** | | **6** | **37** | **21833** | **115** |

**Nothing planned was dropped.**

Six new library modules back the part:

| Module | Lines | Public functions | Test lines | Tests |
|---|---:|---:|---:|---:|
| `differentiation` | 604 | 18 | 340 | 145 |
| `newtoncotes` | 468 | 13 | 380 | 270 |
| `romberg` | 496 | 16 | 356 | 130 |
| `adaptive` | 385 | 10 | 320 | 120 |
| `gaussquad` | 723 | 19 | 413 | 404 |
| `multiquad` | 595 | 18 | 358 | 82 |
| **total** | **3271** | **94** | **2167** | **1151** |

The solutions are [`part09_differentiation_and_integration.md`](../solutions/part09_differentiation_and_integration.md):
5723 lines, 115 solutions, 102 executable code blocks, all of which run.

## B. Source completeness

Part 9 covers **37 of 37** concepts attributed to it, so the part is marked complete in
[`coverage_report.md`](coverage_report.md).

| Source | Chapter | Covered |
|---|---|---|
| Sauer | 5, Differentiation and Integration | 12 of 12 |
| Gupta | 12, Numerical Differentiation | 2 of 2 |
| Gupta | 13, Numerical Integration | 16 of 16 |
| Gupta | 15.6, finite difference approximations | 2 of 2 |
| Trefethen and Bau | L37, Lanczos to Gauss quadrature | 3 of 3 |
| Supplementary | complex step, improper integrals | 2 of 2 |

**Every one of those chapters is now complete across the whole repository**, not merely within
Part 9.

Trefethen and Bau lecture 37 is the source for the Golub-Welsch construction and for the identity
between the Lanczos tridiagonal matrix and the Jacobi matrix of orthogonal polynomials. Both are
used in lesson 65 sections 3 and in solution 5.2, and neither is reproved. **Nothing in this part
is attributed to Trefethen and Bau that is not in it.**

## C. What the measurements changed

The six lessons produced **twenty corrections to code written for this part** and a further
**twenty six in the solutions**, where a measurement contradicted what the exercise expected. The
substantial ones follow.

### Wrong answers that looked right

**`accuracy_order` reported order 1 for every stencil in the library.** The target for the
moment test was written as $k!/(k-m)!$, which is the $m$th derivative of $x^k$ at a general
point. At the origin, which is where the stencil is centred, it is $m!$ when $k = m$ and **zero
otherwise**. With the wrong target every stencil failed at the first test and reported the
minimum. The table now reads 1, 2, 4 and so on, and the symmetric bonus is visible in it.

**The differentiation matrix had no correct digit by 31 nodes, and its residual never said so.**
It was built from `from_taylor`, which solves a Vandermonde system in the offsets. Measured
against exact rational arithmetic on offsets spanning $[-1,1]$, the weights are wrong by a
relative $2.7\times10^{-12}$ at 11 offsets, $0.15$ at 31 and **292 at 41**. The Lagrange route and
Fornberg's recursion are at $10^{-15}$ throughout. The matrix now uses Fornberg.

The failure is a clean instance of forward against backward error. `numpy.linalg.solve` is
backward stable, so the returned weights satisfy the Taylor conditions to $2\times10^{-12}$ at
every width, and the obvious check, that the weights sum to zero, passes for the broken weights
too. **Nothing short of comparison against a stable method finds it.**

**The Gauss-Kronrod nodes were wrong in the second decimal place for every $n \ge 5$.** Laurie's
algorithm has an inner loop that accumulates into a running total; it had been transcribed with
an assignment instead. The result contained the Gauss nodes for $n \le 3$, had positive weights,
summed to 2, and reported a plausible degree, so every structural check passed. It was caught by
comparison against the published QUADPACK 15 point table, which the corrected version now matches
to $10^{-15}$ at every entry.

**The adaptive Richardson divisor was 7 where it should be 15.** The divisor comes from Simpson's
**local** order 5, not its composite order 4, so it is $2^{p} - 1$ and not $2^{p-1} - 1$. The
error was invisible in testing because it makes the estimate $15/7$ too large, which is the safe
direction: the routine subdivides longer than it needs to and passes every accuracy test.
Measured, it costs about 20 percent more evaluations **and** delivers an answer one to two orders
of magnitude worse, because the same divisor is used again in the correction term, where a wrong
coefficient throws away the free two orders.

**`tanh_sinh` stalled at $1.5\times10^{-8}$ for two separate reasons.** Computing the node as
$\text{mid} + \text{half}\tanh u$ cancels to nothing near the endpoints, and a guard written as
`x < b` discarded every node closer to $b$ than machine epsilon, which on $[0,1]$ begins at
$t = 3.2$ and is exactly where the accuracy lives. Both fixed, the rule reaches machine precision.

**A third cause of the same stall is in the caller and cannot be fixed from inside the rule.** An
integrand singular at both ends forms $1 - x$ itself, where $x$ has already rounded to 1. Written
in $x$, $1/\sqrt{x(1-x)}$ never improves past $10^{-8}$ at any level. Given the endpoint
distances, which the rule now returns, the same nodes and weights give exactly zero error.

### Measurements that could not measure what they claimed

**`degree_of_precision` reported 24 for a 21 node Newton-Cotes rule.** Its weights sum to 544 in
absolute value, so the monomials it genuinely misses are missed by less than the rounding those
weights produce. It is now computed in exact rational arithmetic, where there is no tolerance to
choose.

**The same measurement is impossible for Gauss past 16 nodes, and no arithmetic fixes it.** The
rule's relative error on $x^{2n}$, the first monomial it misses, falls from $1.6\times10^{-2}$ at
5 nodes to $1.2\times10^{-8}$ at 16 and $5.8\times10^{-11}$ at 20. Once it is below the tolerance
the walk steps over the true boundary, reporting 41 for a rule whose degree is 39.
`degree_boundary_report` now returns the margin alongside the degree and says which counts can be
believed, and the same treatment is applied to Clenshaw-Curtis in solution 62.5.2.

**A fitted convergence order read 2.8 for a rule of order 2.** Composite trapezoid on Runge's
function does not resolve the peak at its first few panel counts, and a fit over everything above
the roundoff floor averages that pre-asymptotic head into the answer. Simpson read 5.226 and
Boole 6.904 the same way. `convergence` now fits only the run of finest refinements whose local
slopes agree to within 0.25, and reports nan with a stated reason when no such run exists.

**Fitting an algebraic order to a geometrically convergent rule returns a meaningless number.**
`curse_of_dimensionality` originally used Gauss, which on an analytic integrand reports an order
of **21.67**. Dividing that by the dimension demonstrates nothing. It now uses composite Simpson,
whose order is 4 whatever it is given, and the table shows 4.79 per axis at every dimension
against 4.79, 2.40, 1.60 and 1.20 per evaluation.

### Comparisons that were rigged

**The narrow spike demonstration proved the opposite of what it appeared to.** Adaptive Simpson's
first accept decision sees exactly five points: the two ends, the midpoint and the two quarter
points. A spike at $x = 0.5$, which is the obvious place to put it, sits on the first sample and
is found instantly. Moved to $0.4$, the routine returns **zero for a strictly positive integrand**
after nine evaluations. The demonstration now defaults to two fifths and reports the centre sweep
that shows why.

**Monte Carlo's crossover dimension depends on the integrand and not on the dimension alone.** On
a product of cosines a tensor Gauss rule is still ahead at twelve dimensions; on a product of
$|x - \tfrac12|^{1/2}$, which merely has a kink on each axis, Monte Carlo is ahead from three.
`monte_carlo_crossover` now takes the integrand as a parameter and the lesson reports both.

**An odd moment of a symmetric weight is pure cancellation and cannot be judged against a scale
of 1.** Gauss-Hermite at 16 nodes has an exact moment of zero for $x^{31}$ while the terms summing
to it reach $3\times10^{11}$, so the best any arithmetic can do is about $10^{-5}$. Measured
against 1 that reads as catastrophic failure; measured against the terms it is $10^{-16}$.

### Two things that are true and were being reported as failures

**Gauss weights can underflow to zero and it costs nothing.** In exact arithmetic every Gauss
weight is positive. In double precision the outermost Laguerre weights reach $3.5\times10^{-61}$
by 120 nodes and five of them flush to zero. Those nodes sit at $x = 453$, where $e^{-x}$ is far
below anything representable, and the total mass is still right to $10^{-15}$. The positivity
tests now assert non-negativity, with the strict form kept for the finite interval families.

**Half of the classical error constants appeared to disagree with the textbook table.** Every
derived magnitude was right and every sign was flipped. The tables state the error as
$I_{\text{exact}} - I_{\text{rule}}$; the code computed the other difference. The convention is
now recorded in the docstring, since a check that exists to catch transcription errors should not
be silently negated when it catches a convention mismatch instead.

### Corrections in the solutions

Twenty six exercises produced a measurement that contradicted what the exercise expected. Every
one is reported in place rather than replaced with an example that would have worked. The
substantial ones:

- **Exercise 61.4.1 was circular.** `digits_lost` returns $\varepsilon^{p/(p+1)}$, so comparing it
  against $\varepsilon^{p/(p+1)}$ tests nothing. Measured against real stencils, the model is a
  **ceiling** rather than a prediction: the measured best errors are 0.003 to 0.32 of it, and two
  stencils of the same order differ from each other by a factor of 124.
- **Exercise 63.3.2, Euler-Maclaurin anchored at $k = 1$, is wrong by a constant $1.6\times10^{-2}$
  at every $N$**, because the series it produces in place of Euler's constant diverges. Anchoring
  at $k = 10$ and summing the first nine terms directly gives $10^{-15}$ agreement at
  $N = 10^{12}$.
- **Exercise 63.4.3's threshold is sharp and checkable.** For $\sin(wx)$ the Euler-Maclaurin
  series converges when $hw < 2\pi$ and diverges when $hw > 2\pi$, confirmed at seven points. On
  the lesson's own examples $hw \le 1$, so **the divergence that makes the series asymptotic is
  invisible on the easy cases.**
- **Exercise 65.4.2 separates two rates that are usually quoted as one.** An **endpoint**
  algebraic singularity gives Gauss a rate of $2a+2$; an **interior** one gives only $a+1$. Gauss
  nodes cluster at the ends with spacing $O(n^{-2})$ and are sparsest in the middle, so the
  ordering is the reverse of the naive expectation.
- **Exercise 66.3.1's comparison had to be redone at equal cost.** Gauss-Laguerre on
  $e^{-x}\cos x$ reaches $6\times10^{-17}$ with 25 points, thirteen orders ahead of the
  transformed route. On an integrand with no exponential factor it returns $2.7\times10^{17}$
  when the answer is $0.5$.
- **Three exercises end in a negative result and say so.** The sparse grid of 66.3.4 loses to a
  full tensor grid at every size reachable here, because the integrand is analytic and the tensor
  rule converges geometrically. The $N/\log N$ scaling of 66.5.2 is unobservable in double
  precision, because the rule reaches exact zero by level 3. The ridge counterexample of 66.5.3
  does not separate from the separable case at the levels a direct construction can reach.

**Solution 64.5.3 records a defect introduced while answering it.** A roundoff detector written to
test whether the local estimate had stopped shrinking fired on an unresolved peak, accepted an
interval containing the whole feature, and returned an answer with 100 percent error while
reporting a small stall count. Rewritten to test against the roundoff floor of the interval's own
value, it fires exactly where the tolerance is missed, and it also **improves** the answer: the
same integrand that missed $10^{-12}$ by a factor of 215 after several million evaluations now
misses by a factor of 7 after sixteen thousand.

## D. Two defects in the audit tooling

Both were found by checking the coverage flags against the repository rather than reading them.

**The `tested` flag was decided by a table listing lessons 01 to 08.** Every lesson from 09
onward reported `tested: False` regardless of how many tests backed it, so the repository read
**49 of 507** when the true figure is 384. The flag is now derived from the modules each lesson
imports, and a module counts as tested if any test file exercises it, since `pivoting` is tested
inside `test_lu.py` and that is a reasonable place for it.

**The `experimented` flag required an inline PNG in the notebook.** Every lesson in Parts 7, 8 and
9 saves its figure to `figures/` and closes it, which produces no inline output, so all three
parts reported `experimented: False` for figures that exist on disk and are linked from the built
markdown. The flag now accepts either.

After both repairs, `covered`, `tested`, `verified` and `experimented` all read **384 of 507**,
one for every concept in a lesson that exists.

**A third defect was introduced by the first repair and caught immediately.** The import pattern
used `[\w,\s]+`, and `\s` matches a newline, so `from nalib import linalg` followed by a line
beginning `C = ...` captured `C` as a module name and reported lesson 04 as untested. The pattern
now stops at the end of its line.

## E. What is not claimed

- **The complex step is exact for the functions tested and is not a general derivative method.**
  It needs $f$ holomorphic at the point and real on the real axis. `abs`, `real` and `conj` return
  exactly zero, silently. `sqrt(z**2)` and `maximum` work away from their kinks, which is
  measured rather than assumed.

- **Spectral differentiation's minimum is measured on the integrands tested and is not a
  universal node count.** On $\sin$ it is at 17 nodes; on $|x|^{1.5}$ the error is still falling at
  257. The minimum moves right as the integrand gets rougher, which is the opposite of the usual
  intuition and is measured in solution 61.4.2.

- **No claim is made that adaptive quadrature meets its tolerance.** It does on four of the five
  integrands tested and fails on the fifth by a factor of 215, with no warning, because the
  tolerance was below the roundoff floor of its own summation.

- **The Gauss against Clenshaw-Curtis comparison is reported as integrand dependent, and the
  folklore that they are comparable is shown to fail.** On $\sin(3x)$ Clenshaw-Curtis wins at
  every size; on $e^x$ at 9 points Gauss wins by 7680; on a pole at $1.1$ Gauss pulls ahead
  without limit, reaching 2450 by 33 points.

- **The Kronrod extension is verified to exist and behave for the Legendre weight only.** That is
  a property of that weight, not a theorem, and solution 65.5.1 says so.

- **Lesson 66's dimension results are for the two integrand families tested.** The crossover
  dimension, the sparse grid comparison and the quasi Monte Carlo advantage all depend on the
  integrand, and each is reported with the family that produced it.

- **The tanh-sinh rule's indifference to singularity strength is wide and not unconditional.** It
  needs two levels for $x^{-p}$ at every $p$ from 0.25 to 0.95 and fails outright at $p = 0.99$,
  where the transformed integrand decays a hundred times more slowly and the tail cut discards too
  much.

## F. Reproducing every number here

```
python verification/run_all.py
```

That rebuilds and executes all 66 notebooks, runs the full test suite, re-runs the identity
checks, regenerates the coverage checklist, and runs the pairing, link, placeholder, generality
and control character sweeps. It takes a few minutes and prints a pass or fail for each.
