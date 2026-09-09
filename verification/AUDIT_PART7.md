# Audit: Part 7, Interpolation

A part-level audit. [FINAL_AUDIT.md](FINAL_AUDIT.md) carries the current overall position and
links to every part-level audit, so that list is maintained in one place rather than repeated
here and going stale each time a part lands.

Everything below is measured from the repository by [`run_all.py`](run_all.py). No number in
this document was typed in by hand.

---

## Verdict

**Part 7 passes.** All ten lessons exist, all are executed, tested, numerically verified and
fully solved, with no placeholders, no unpaired files and no hardcoded dimensions.

| Check | Result |
|---|---|
| Lessons written | **10 of 10** |
| Notebooks executed end to end | **10 of 10 pass** (53 of 53 across the repository) |
| Automated tests | **4849 of 4849 pass** (1074 new) |
| Assertions inside the Part 7 lessons | **76, all passing** |
| Concepts covered from the sources | **59 new, 320 of 507 total** |
| Concepts outstanding in Part 7 | **0** |
| Worked exercise solutions | **170**, one for every exercise |
| Generality scan | **0 candidates across 256 files** |
| Mangled escape scan | **0 control characters across the repository** |

---

## A. Curriculum completeness

Part 7 was planned as 10 lessons. **All ten exist.**

| Lesson | Title | Figures | Assertions | Words | Exercises |
|---|---|---:|---:|---:|---:|
| 44 | Polynomial Interpolation and Its Five Forms | 1 | 5 | 2917 | 17 |
| 45 | Divided Differences | 1 | 8 | 2862 | 17 |
| 46 | Interpolation Error and the Runge Phenomenon | 1 | 8 | 2866 | 17 |
| 47 | Chebyshev Interpolation | 1 | 10 | 3340 | 17 |
| 48 | Finite Difference Operators and Tables | 1 | 8 | 2942 | 17 |
| 49 | The Equal Interval Interpolation Formulas | 1 | 6 | 3018 | 17 |
| 50 | Hermite and Piecewise Interpolation | 1 | 9 | 3060 | 17 |
| 51 | Cubic Splines | 1 | 7 | 2978 | 17 |
| 52 | Bezier and B-spline Curves | 1 | 8 | 3460 | 17 |
| 53 | Bivariate Interpolation, and the Wall | 1 | 7 | 2656 | 17 |
| **total** | | **10** | **76** | **30099** | **170** |

**Nothing planned was dropped.**

Ten new library modules back the part, `interp`, `divdiff`, `interperror`, `chebyshev`,
`finitediff`, `equalinterval`, `hermite`, `splines`, `bezier` and `bivariate`, at 3387 lines,
with 2314 lines of tests against them.

## B. Source completeness

Part 7 covers **59 of 59** concepts attributed to it, so the part is marked complete in
[`coverage_report.md`](coverage_report.md).

The classical equal-interval material, Gregory-Newton through Everett and Steffensen, comes from
the Indian textbook tradition, source S2 chapter 10, and it is the only place in this course
where those formulas appear. They are treated as history that still teaches something rather
than as a live method, and lesson 49's exercise 5.3 says what replaced them and why.

Trefethen and Bau contributes the potential theory framing of the Runge phenomenon and the
Bernstein ellipse, and it is cited for those rather than reproved. Nothing in this part is
attributed to Trefethen and Bau that is not in it.

## C. What the measurements changed

Ten lessons produced eight corrections to claims that a textbook statement or the exercise
prompt had led the first draft to expect, and six defects in the library.

**de Casteljau is not more accurate than the Bernstein sum, at any practical degree.** The
folklore says it is. Measured, the two agree to $4\times10^{-16}$ at degree 40 and
$8.9\times10^{-16}$ at degree 200, where the largest binomial coefficient is $9\times10^{58}$.
On $[0,1]$ the Bernstein sum is a convex combination, so nothing cancels and no intermediate
quantity exceeds the answer. The two do differ **outside** $[0,1]$, by $6.0\times10^{-8}$ at
degree 60 and $t = 1.2$, where the weights alternate in sign, and that is a case nobody makes
the claim about. The Bernstein form's real failure is loud rather than quiet: `OverflowError` at
degree 1030, when the binomial coefficient stops fitting in a double.

**The classical rule for choosing an equal-interval formula is right at the ends and wrong in
the middle.** Measured at 18 probe positions across a 13 node table, the rule picks the actual
winner **6 of 18 times**, at every order tested. The failures are systematic and all in the same
place: in the middle third, where the rule says Stirling or Bessel, the measurement says Everett
or Steffensen. The rule is nevertheless worth following, because the automatic choice built on
it beats **every** single fixed formula at **every** order, reaching $1.9\times10^{-3}$ at order
2 against the best fixed formula's $3.8\times10^{-3}$ and $4.9\times10^{-4}$ at order 6 against
Gregory-Newton forward's $4.7\times10^{-3}$. The reason is that the maximum over the interval is
attained at the ends, where the rule is right, so its failures in the middle never surface.

**Compensated summation buys 0.2 digits at best in a divided difference table, and nothing at
the orders where it is needed.** Measured across orders 4 to 16, the digits recovered are 0.0,
0.1, 0.2, -0.1, -0.0. The reason is that the loss is not in the summation. The recursion forms
$(a-b)/h$ where $a$ and $b$ agree to many digits, so by Sterbenz's lemma the **subtraction is
exact** and the information was already gone before it happened. Capturing the rounding of an
exact operation captures nothing.

**Lagrange does not always beat Hermite at equal data.** Hermite at $n$ nodes and Lagrange at
$2n$ both give degree $2n-1$, and the textbook expectation is that Lagrange wins. Measured
across four functions and four sizes, Lagrange wins only while approximation error dominates.
Hermite wins at machine precision, and on Runge's function it wins outright.

**Chebyshev-Lobatto points are asymptotically optimal for the Lebesgue constant, and the
measurement shows the convergence rather than assuming it.** Erdos proved
$\Lambda_n \ge \frac2\pi\log n + 0.9625$ for every node family. The implied constant for
Chebyshev-Lobatto climbs 0.8960, 0.9410, 0.9561, 0.9604, 0.9615 as $n$ runs 10 to 600, toward
that bound. A single measurement at $n = 10$ would have suggested the points miss the bound by 7
percent, which is the wrong conclusion.

**Chebyshev knots are the worst of three choices for a spline.** On $1/(1+400t^2)$ at 65 knots,
Chebyshev gives $1.99\times10^{-2}$, uniform gives $3.74\times10^{-3}$, and adaptively clustered
knots give $5.37\times10^{-5}$. Chebyshev clusters at the **ends**, which is right for
polynomial interpolation, where the node polynomial's growth is an end effect, and wrong for
splines, which have no node polynomial and need knots where the function is hard. Carrying the
rule across from lesson 47 to lesson 51 would have been an error, and the measurement catches it.

**Non-uniform knots never become a conditioning problem for the spline moment system.** The raw
condition number tracks the ratio of largest to smallest interval, reaching $1.6\times10^{13}$
at a ratio of $10^{14}$, and that is entirely row scaling: after row equilibration it runs 2.995
to 3.105 across fourteen orders of grading, and the spline interpolates its own data to
$10^{-16}$ throughout. With the ratio applied to two adjacent intervals instead, leaving the
rest uniform, the condition number saturates at 44 and the sup error does not move in the fourth
digit across twelve orders. What degrades under grading is approximation, set by the **largest**
interval, and the measured sup error tracks the $\frac{5}{384}h^4\|f^{(4)}\|$ bound.

**The natural spline's apparent order of 5 to 9 in the interior is boundary pollution decaying,
not superconvergence.** Measured over the middle 80 percent of $[0,1]$ on $\exp$, the fitted
order reads 5.460, 7.748, 8.927 at 32, 64 and 128 intervals, then 4.014, 3.994, 3.907 at 256,
512 and 1024. The end condition's $O(1)$ moment error decays geometrically in knot index, at a
ratio near $2-\sqrt3$, so while it still exceeds the interior $O(h^4)$ it inflates the fit.
Stopping the refinement at 128 would have produced a claim of order 9.

**Six library defects, found and fixed.**

- `nalib.hermite.hermite_newton` mixed two table conventions, placing the confluent entry at the
  Burden and Faires diagonal position while the recursion used the top-row convention. The
  interpolant was wrong by 0.718 on a two node test. Placing the entry where the recursion reads
  it fixes it to machine precision.
- `nalib.splines.smoothness_report` sampled at $x \pm \epsilon$ and so measured
  $2\epsilon S'$ rather than a jump. Every "jump" read about $4\times10^{-7}$, which is exactly
  what a small genuine jump would look like. It now evaluates each piece **at** the knot using
  that piece's own cubic.
- `nalib.splines.minimum_curvature` compared the natural spline against competitors formed by
  perturbing its moments. Those are not $C^1$, so they are not admissible in the theorem's
  competition class, and some of them have lower bending energy. The competitors are now the
  splines with other end moments, every one of which is a genuine $C^2$ interpolant of the same
  data.
- `nalib.finitediff.locate_a_bad_value` clipped its comparison window at the start of the
  difference table without shifting the expected binomial pattern with it, so it rejected correct
  answers at indices 2 and 3. The pattern is now aligned to the peak's offset, which may be
  negative.
- `nalib.finitediff.interrelations` checked $\Delta = \nabla E$ with the two sides offset by one
  element, reporting a discrepancy of 3.84 for an identity that holds exactly.
- `nalib.interp.evaluate_barycentric` divided by an exactly zero denominator at 64 equally spaced
  nodes, where the weights span $10^{18}$ and the sum cancels completely. It now returns `nan`
  with the warning suppressed, and the docstring says why, rather than returning an infinity that
  looks like an answer.

**One hole in this audit's own tooling, found and fixed, for the second time.** `run_all.py`'s
generality check built its own list of files to scan, and that list omitted the worked solutions,
the built notebooks and the built markdown lessons. It reported 0 candidates across 142 files
while the standalone scanner reported **1 across 256**: an `np.eye(3)` in this part's own
solutions file, in the barycentric triangle code of solution 53.3.3. The Part 6 audit had already
patched the same list once, to add the tests. Rather than patch it a third time, `run_all.py` now
calls `check_generality.collect_targets`, so the two scanners share one file list as well as one
rule and can no longer disagree. The offending line now reads `np.eye(tri.shape[0])`, which is
what the rule asks for even though a triangle has three vertices by definition.

## D. What is deliberately not claimed

- **The natural end condition never beats the others, it only stops losing.** On $\sin$ over
  $[0,\pi]$ and $[0,2\pi]$, where $f''$ vanishes at both ends, natural matches clamped and
  not-a-knot to every digit reported, $1.512\times10^{-8}$ against $1.512\times10^{-8}$. Taking a
  `min` over that row would have reported "natural wins", which is tie breaking, not a result.
  The claim made is that natural recovers order 4 there, which is what the order columns show.

- **The B-spline basis is not better conditioned than the moment formulation, nor worse.** The
  raw collocation condition number grows like $n^2$, reaching $5.4\times10^5$ at 129 nodes, and
  that growth is entirely the two end rows being divided by $\epsilon^2$: changing the difference
  step changes the number. After row equilibration it is 3.21 against the moment matrix's 3.00,
  both bounded independently of $n$. The reasons to prefer B-splines are local support and the
  hull bound, and the audit does not let a scaling artefact be quoted as a third.

- **The order of a linear interpolant on a Delaunay triangulation is not measurable on random
  points.** With random points the worst triangle's aspect ratio grows from 624 to
  $6.5\times10^4$ as points are added, and the fitted maximum-norm order swings 1.64, 3.17, 1.84
  even taking the median over nine seeds. On a perturbed grid, where the aspect ratio stays
  between 5.6 and 7.5, the rms order reads 2.157, 2.093, 2.030. The order 2 is claimed for the
  controlled case and explicitly not for the uncontrolled one.

- **RBF accuracy continues to improve while the condition number passes $10^{18}$.** At 256
  gridded centres with a Gaussian kernel the matrix condition number is $4.05\times10^{18}$ and
  the interpolation error is $2.29\times10^{-11}$, eight orders better than at 64 centres. The
  ill conditioning is a property of the basis, not of the problem, and stopping on the condition
  number alone would have thrown that accuracy away. The limit is real and is watched by the
  **residual**: it stays at $10^{-16}$ up to shape 0.4, reaches $7.6\times10^{-11}$ at 0.8, and
  $5.2\times10^{-2}$ at 6.4, past which the result is not an interpolant. The rule recorded is
  to watch the residual, not the condition number.

- **The measured Monte Carlo crossover is at $k = 9$, and the model says 8.** For Simpson's rule
  at order 4 the count model predicts the crossover where $k/p = 2$. The measurement puts it one
  dimension later, because Monte Carlo's constant is $\sigma_f^2$ and Simpson's is small on this
  very smooth integrand. The **slope** is the model's and that is what is claimed; the exact
  crossover dimension is specific to the integrand and to the tolerance, and the solution says so.

- **A polynomial Bezier curve reaches $7\times10^{-15}$ on a quarter circle at degree 12.** The
  theorem that no polynomial curve **is** a circle is proved and is exact, but quoting it as a
  practical accuracy advantage for NURBS would be dishonest. The advantages claimed are
  exactness under composition, degree 2 instead of 12, closure under projective transformation,
  and one representation for the whole conic family.

- **The excerpt in solution 44.1.3 is library source, not a runnable program.** It is four lines
  lifted from `nalib.interp.evaluate_barycentric` to show the exact-hit guard, and it references
  names bound in the surrounding function. Every other python block in the Part 7 solutions runs
  standalone from the repository root; that one is marked by its prose as an excerpt.

- **The barycentric formula returns `nan` at 64 equally spaced nodes and this is not repaired.**
  The denominator cancels to exactly zero because the weights span $10^{18}$. That is the
  correct answer to report: the value is not determined by the data in double precision, and
  returning a large finite number instead would be worse. Equally spaced nodes at that count are
  outside the method's usable range, which is lesson 47's whole point.

- **Timing does not appear in this part.** Every comparison is of accuracy, operation counts or
  condition numbers, all of which are machine independent. Where a cost claim is made, for
  instance that adaptive knots beat uniform ones by two orders, it is a claim about error at a
  fixed knot count and not about wall clock time.
