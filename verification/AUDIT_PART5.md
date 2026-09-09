# Audit: Part 5, Orthogonality, QR and Least Squares

A part-level audit. [FINAL_AUDIT.md](FINAL_AUDIT.md) carries the current overall position and
links to every part-level audit, so that list is maintained in one place rather than repeated
here and going stale each time a part lands.

Everything below is measured from the repository by [`run_all.py`](run_all.py). No number in
this document was typed in by hand.

---

## Verdict

**Part 5 passes.** All six lessons exist, all are executed, tested, numerically verified and
fully solved, with no placeholders, no unpaired files and no hardcoded dimensions.

| Check | Result |
|---|---|
| Lessons written | **6 of 6** |
| Notebooks executed end to end | **6 of 6 pass** (34 of 34 across the repository) |
| Automated tests | **3026 of 3026 pass** (714 new) |
| Assertions inside the Part 5 lessons | **50, all passing** |
| Concepts covered from the sources | **35 new, 214 of 507 total** |
| Concepts outstanding in Part 5 | **0** |
| Worked exercise solutions | **102**, one for every exercise |
| Generality scan | **0 candidates across 159 files** |
| Mangled escape scan | **0 control characters across the repository** |

---

## A. Curriculum completeness

Part 5 was planned as 6 lessons. **All six exist.**

| Lesson | Title | Figures | Assertions | Words | Exercises |
|---|---|---:|---:|---:|---:|
| 29 | Least Squares and the Normal Equations | 1 | 9 | 2706 | 17 |
| 30 | Gram-Schmidt and QR | 1 | 6 | 2511 | 17 |
| 31 | Householder and Givens QR | 1 | 6 | 2546 | 17 |
| 32 | Solving Least Squares in Practice | 1 | 4 | 2477 | 17 |
| 33 | Rank Revealing QR and Total Least Squares | 1 | 17 | 4108 | 17 |
| 34 | Nonlinear Least Squares | 1 | 8 | 3329 | 17 |
| **total** | | **6** | **50** | **17677** | **102** |

**Nothing planned was dropped.**

## B. Source completeness

| Source | Chapter | Concepts covered by lessons 29 to 34 |
|---|---|---:|
| S1 Sauer | 4 Least Squares | 14 |
| S2 Gupta | 11 Splines and Curve Fitting | 3 |
| S3 Trefethen and Bau | L8 Gram-Schmidt | 2 |
| S3 Trefethen and Bau | L7 QR Factorization | 1 |
| S3 Trefethen and Bau | L10 Householder | 3 |
| S3 Trefethen and Bau | L16 Stability of Householder | 2 |
| S3 Trefethen and Bau | L11 Least Squares | 1 |
| S3 Trefethen and Bau | L18 Conditioning of Least Squares | 2 |
| S3 Trefethen and Bau | L19 Stability of Least Squares Algorithms | 2 |
| S3 Trefethen and Bau | L4, L11 | 1 |
| S3 Trefethen and Bau | L35 | 1 |
| SUP supplementary | regularization, rank revealing QR, total least squares | 3 |
| | **total** | **35** |

**Trefethen and Bau supplies 15 of the 35**, which is the pattern the original instruction asked
for: the numerical linear algebra parts lean on it hardest. Lectures 7, 8, 10, 11, 16, 18 and 19
are the spine of Part 5, and lectures 16, 18 and 19 in particular are what lessons 31 and 32 are
built on.

**Sauer chapter 4 is now complete**, all 14 concepts, with 4.5 (models with nonlinear parameters,
Gauss-Newton, Levenberg-Marquardt) and RC4 (GPS) landing in lesson 34.

**Nothing is outstanding in Part 5.**

**One thing this table deliberately does not claim.** Lesson 32 uses the SVD heavily, but the
checklist rows for the SVD itself (Trefethen L4 and L5: conditioning through singular values,
minimum norm solutions via the pseudoinverse, Weyl perturbation) are assigned to **lessons 41
and 43**, in Part 6, where the SVD is developed rather than used. They are still counted as
outstanding. Marking them here because lesson 32 touches them would be exactly the kind of
inflation this audit exists to prevent.

## C. Numerical linear algebra depth

| Requested depth | Where, and what is measured |
|---|---|
| Orthogonal projectors as objects, not formulas | Lesson 29 section 3: $P^2 = P$, $P = P^T$, $\|P\|_2 = 1$ verified at four shapes to $4\times10^{-16}$, and the lesson then explains why $P$ is never formed |
| Conditioning squared by the normal equations | Lesson 29 section 5: $\kappa(A^TA)/\kappa(A)^2 = 1$ to six digits across five decades, with the fitted error slopes **1.968** and **0.926**, ratio **2.13** against a theoretical 2 |
| Triangular orthogonalization against orthogonal triangularization | Lesson 30 section 9 and lesson 31 section 1, with the measured consequence: Gram-Schmidt loses orthogonality like $\kappa^{1.86}$ or $\kappa^{0.98}$, Householder loses none at any $\kappa$ |
| Backward stability of Householder QR | Lesson 31 section 8 and solution 31.4.1: the error in $Q$ grows to **0.613** at $\kappa = 10^{18}$ while $\|A - QR\|$ stays at $8\times10^{-16}$ and $\|Q^TQ-I\|$ at $9\times10^{-16}$ |
| The four condition numbers of a least squares problem | Lesson 32 section 1 and solution 32.4.1: measured across $\theta$ from 0.001 to 89.9 degrees, spanning $10^{6}$ to $7\times10^{8}$ on one axis and 1 to 695 on another, **on the same matrix** |
| The pseudoinverse and the Penrose conditions | Lesson 32 section 3: all four residuals below $10^{-13}$ for tall, wide, full rank and rank deficient, with the proof of uniqueness in solution 32.2.2 |
| Regularization as a change of problem | Lesson 32 sections 6 and 7, and solutions 32.3.2, 32.4.3, 32.5.1: the filter factors, the L-curve, GCV, and Tikhonov as the maximum a posteriori estimate with $\lambda = \sigma/\tau$ confirmed to a factor of 0.86 to 1.44 |
| Rank revealing factorizations, and their limits | Lesson 33 sections 2 to 4: pivoted QR built from scratch, and Kahan's matrix defeating it with **zero swaps** at every size while understating $\kappa$ by up to $3\times10^{14}$. Solution 33.3.2 implements strong RRQR, which defeats Kahan in **2 swaps** at 1.2 times the cost |
| Total least squares as a different problem | Lesson 33 sections 6 to 8: the smallest singular vector of $[A\mid\mathbf{b}]$, the correction equal to $\sigma_{\min}$ exactly, the attenuation bias confirmed to three decimals, and the bias surviving a 400-fold increase in sample size |
| Nonlinear least squares as the linear theory iterated | Lesson 34 sections 2 and 3: the exact Hessian $J^TJ+S$, and the measured correspondence between $\|S\|/\|J^TJ\|$ and the iteration count, with a **held-out prediction of 16 against an actual 17** in solution 34.4.2 |

## D. Computational completeness

**Four library modules were written for this part.**

| Module | Lines | Public functions |
|---|---:|---:|
| `src/nalib/leastsquares.py` | 632 | 24 |
| `src/nalib/qr.py` | 432 | 14 |
| `src/nalib/rrqr.py` | 436 | 11 |
| `src/nalib/nlls.py` | 512 | 9 |

Every important algorithm is implemented from scratch and then checked against both a trusted
library and a defining identity:

| From scratch | Checked against |
|---|---|
| Classical, modified and reorthogonalized Gram-Schmidt | `numpy.linalg.qr`, and $\|Q^TQ-I\|$, which the library cannot supply |
| Householder QR, full and compact | a factorization built from known factors $A = Q_0R_0$, so no higher precision reference is needed |
| Givens QR, with a rotation counter | Householder on the same matrix, agreeing to a factor of two at every $\kappa$ |
| Normal equations, Cholesky normal equations, QR and SVD solvers | each other and $\mathbf{x}_{\text{true}}$, across $\kappa$ from $10$ to $10^{16}$ |
| The pseudoinverse | all four Penrose conditions |
| Tikhonov, truncated SVD, the L-curve | the filter factor identity, and each other on spectra with and without a gap |
| Column pivoted QR, with the LINPACK norm downdate | `A[:, piv] = QR` to $10^{-13}$, a non-increasing diagonal by construction, and the SVD's rank on matrices where both agree |
| Total least squares | the Penrose-free identity that the correction equals $\sigma_{\min}([A\mid\mathbf{b}])$ exactly, plus a second closed form derived independently |
| Gauss-Newton, Levenberg-Marquardt, full Newton | each other, a finite difference Jacobian, and analytic Hessians checked against finite differences of the analytic Jacobians |

**Tests.** 206 test functions across the four Part 5 test modules, inside a suite of 3026:

| Test module | Test functions |
|---|---:|
| `tests/test_leastsquares.py` | 82 |
| `tests/test_qr.py` | 51 |
| `tests/test_rrqr.py` | 38 |
| `tests/test_nlls.py` | 35 |

**Solutions.** [`solutions/part05_least_squares_and_qr.md`](../solutions/part05_least_squares_and_qr.md),
3597 lines, **102 worked solutions for 102 exercises**, with 21 runnable code blocks.

**The code in those solutions is executed, not merely printed.** Two harnesses extract every
`python` block, run it, and exercise what it defines: at five shapes including the square case
$7\times7$ and the single column case $5\times1$, at three quadrature sizes, three regularization
sizes, three Kahan sizes, four column-deletion positions and three data sizes for the robust
fit, and they confirm that every input guard raises. All checks pass.

## E. Defects this process caught

Listed because each one was found by a check, not by reading.

| Defect | How it showed up | Fix |
|---|---|---|
| **Gram-Schmidt's dependency guard never fired** | an exactly dependent column leaves a residual of about $u\|A\|$, not zero, so `== 0.0` never triggered and the routine returned a $Q$ with orthogonality error **0.92** while reporting success | the standard relative threshold $\max(m,n)\,u\,\|A\|_2$ |
| **`angle_to_range` used `arccos`** | at a relative residual below $10^{-8}$ it returned **exactly 0.0**, a 100 percent error, because the derivative of arccos is infinite where its argument sits | `arctan2(\|r\|, \|Pb\|)`, exact to $4\times10^{-16}$ |
| **`measure_sensitivity` normalised by $\|r\|$** | unbounded when the fit is nearly exact, for reasons having nothing to do with conditioning | normalise by $\|b\|$ |
| **`numpy.linalg.qr` rejects `longdouble`** | lesson 31's high precision reference could not be computed at all | build $A = Q_0R_0$ from known factors instead, which needs no reference |
| **Tikhonov's limit test asserted monotone improvement** | in floating point the error falls to $2.7\times10^{-8}$ at $\lambda = 10^{-3}$ and returns to $2\times10^{5}$ by $10^{-10}$, because the singular values that should be zero are about $u\|A\|$ | assert the approach **and** the divergence, and add a test that the useful range is bounded below by about $\sqrt{u\|A\|}$ |
| **`rank_gap` reported the gap wherever the first exact zero fell** | on a rank one $20\times20$ matrix the diagonal below the rank is roundoff with exact zeros scattered through it, and a ratio against an exact zero is infinite. The gap was reported at index 7 instead of 1 | restrict the search to entries above $n\,u\,\lvert r_{11}\rvert$: a ratio between two roundoff values is noise divided by noise |
| **The TLS existence test checked only the last singular vector entry** | with $\mathbf{b}$ orthogonal to $\operatorname{range}(A)$ that entry is zero exactly and $10^{-16}$ numerically, so it passed and the routine divided by it, returning components of order $10^{14}$ | add Golub and Van Loan's gap criterion $\sigma_n(A) > \sigma_{n+1}(C)$ alongside it |
| **`gauss_newton` stopped on the unscaled gradient** | $\|J^T\mathbf{r}\|$ carries the units of $J$ times $\mathbf{r}$, so the same problem in different units fails a test it used to pass. At the true minimum of the large residual problem it read $1.0\times10^{-8}$, which looks like failure, against a scaled $4.9\times10^{-11}$ | divide by $\|J\|\,\|\mathbf{r}\|$, and add `ftol` and `xtol`, because a nonzero residual iteration limit cycles rather than contracting |
| **Solution 33.5.3 unscaled `scaled_tls` the wrong way** | $\mathbf{y} = \mathbf{x}/\gamma$ turns the scaled problem into plain TLS, so the answer is $\gamma\mathbf{y}$ and not $\mathbf{y}/\gamma$. **$\gamma = 1$ hides it**, which is why it survived the first check | corrected, and the limits renamed: $\gamma\to\infty$ is ordinary least squares, $\gamma\to0$ is **data** least squares, and the text had those backwards too |
| **The truncation against Tikhonov test used a penalty inside the gap** | the filter factor for a tiny $\sigma$ is $\sigma/\lambda^2$, so $\lambda$ must exceed $\sqrt{\sigma}$ to suppress rather than amplify | put $\lambda$ at the geometric middle, and assert that a penalty inside the gap but below $\sqrt{\sigma}$ does **not** match |
| **37 mangled escapes across 10 files** | a shell heredoc had interpreted `\times` as a tab plus `imes` and `\approx` as a BEL plus `pprox`; the files rendered, and every other check passed | all 37 repaired, and a **new step 9 of 9** in `run_all.py` sweeps for control characters so it cannot recur silently |
| **A `\b` word boundary was destroyed in the generality scanner itself** | one of the 37, and the worst placed: it widened the scanner's own regex | repaired; the scanner still reports 0 candidates, so nothing had been masked |
| **The notebook report counted unwritten lessons as executed** | "34 executed, 32 passed, 0 failed" on a repository holding 32 notebooks | `NO_SOURCE` entries are now excluded from the executed count and listed separately as outstanding |

## F. Claims corrected against measurement

Each of these was the expected answer before it was measured, and each is wrong.

| Expected | Measured |
|---|---|
| Chebyshev sample spacing improves the conditioning of a polynomial fit | It does **not**. It is slightly *worse* than even spacing at every degree tested, by a factor of 0.85 at degree 20. **The basis is the whole story**: Legendre against monomial is $\kappa = 7.76$ against $1.96\times10^{7}$, a factor of $2.5\times10^{6}$ |
| Seminormal equations improve on the normal equations by not forming $A^TA$ | Plain seminormal is **no better**, and at $\kappa = 10^{10}$ it is three orders of magnitude **worse**. One step of refinement changes it completely, matching QR up to $\kappa = 10^{8}$ |
| Reordering the columns rescues classical Gram-Schmidt | It matters by a factor of about 20 and no more, and **no ordering rescues it**. Sorting by column norm, the obvious heuristic, is 1.5 times *worse* than the natural order at $\kappa = 10^{8}$ |
| Householder QR is slower than Gram-Schmidt because it is more stable | It is **faster at every size tested**, by a factor of 1.3 to 3, and the flop count already says so: $4n^3/3$ against $2n^3$ |
| The Givens against Householder crossover is near the flop-count prediction $p \approx n/3$ | It is at **bandwidth 7** out of 200, because Givens is $O(n^2)$ tiny two-row updates while Householder's work goes to BLAS. Householder's time does not vary with bandwidth **at all** |
| The SVD's default `rcond` is a safe choice | On one $\kappa = 10^{14}$ problem it was **2600 times worse** than keeping everything. On a noisier problem it would have been far too generous. The threshold trades bias against variance and cannot see the noise level |
| A general regularizer obviously beats the identity penalty | On the obvious test problem all three tie within 6 percent, because the truth was both small and smooth. **The demonstration needed a truth with a large mean**: at offset 50 the difference penalties are 3.3 times better, because $D_1$ annihilates a constant |
| MGS is unusable for least squares because its $Q$ is only orthogonal to $u\kappa$ | Orthogonalizing $[A\mid\mathbf{b}]$ **matches Householder within a factor of 3** at every usable $\kappa$, with the same bad $Q$. "The $Q$ is bad" and "the answer is bad" are different claims |
| Plain QR fails to reveal the rank of an exactly rank deficient matrix | It does **not** fail there: any ordering leaves exactly $r$ large diagonal entries. What it fails at is putting them **in order**, so the "walk down until it drops" rule reports rank 1 for a rank 4 matrix. The first draft of lesson 33 demonstrated the wrong thing and was rebuilt |
| Greedy subset selection degrades as the columns become more correlated | Measured across five correlation levels: it is optimal about **2 times in 50 at every level**, with a median $\sigma_{\min}$ ratio flat at 0.90. The correlation is not what drives it |
| Selecting on $\sigma_{\min}$ and on the projection residual are two approximations to one answer | They agree in **1 to 4 trials out of 50**. They are answers to two different questions, and which you want depends on whether you will solve with the columns or summarise with them |
| Levenberg-Marquardt is more robust than Gauss-Newton | Measured over 676 starting points: Gauss-Newton reached the best minimum from **45 percent** and Levenberg-Marquardt from **20 percent**, and neither blew up. Damping buys reliable *convergence* and costs basin size |
| Keeping the exact Hessian is better than dropping $S$ | Full Newton took **fewer** iterations at every residual level and landed on a **worse** point every time, because $J^TJ+S$ was indefinite at 144 of 200 sampled points while $J^TJ$ cannot be |
| A trust region method needs only a gradient stopping test | Without `ftol` and `xtol` it ran to the 300 iteration limit on **four of five** problems with the right answer already in hand |

## Statistics for Part 5

Measured by `run_all.py` at the time of writing.

| Quantity | Value |
|---|---:|
| Lessons written in Part 5 | 6 of 6 |
| Notebooks passing, repository wide | 34 of 34 |
| Notebooks failing | 0 |
| Tests passing | 3026 |
| Tests failing | 0 |
| Numerical identities re-verified | 13 of 13 |
| Assertions inside notebooks, repository wide | 319 |
| Concepts covered | 214 of 507 |
| Lessons built, repository wide | 34 of 96 |
| Pairing problems | 0 |
| Broken links | 0 |
| Placeholders | 0 |
| Generality candidates | 0 |
| Control characters | 0 |
| Solution files | 5 |
| Library modules | 26 |
| Test modules | 25 |

## What is deliberately not claimed

- **The SVD is used in Part 5 but not developed.** Lessons 32 and 33 apply it, and lesson 33
  uses the Eckart-Young theorem on credit; lessons 41 to 43 build both. The corresponding
  checklist rows stay uncovered and are counted against Part 6.
- **Rank revealing QR is presented as a heuristic, and the audit does not claim otherwise.**
  Kahan's matrix defeats plain pivoting at every size tested, and the strong version in solution
  33.3.2 fixes that case without a proof that it fixes every case; Gu and Eisenstat's bound is
  cited, not verified here.
- **Neither nonlinear method searches globally**, and no claim is made that either finds the
  best minimum. The basin measurements in solution 34.4.1 are on one problem, one grid.
- **The four condition numbers were measured on one matrix family**, graded designs at
  $30\times6$ and $60\times8$. The formulas are proved in the solutions and confirmed there;
  a wider sweep of matrix families is not claimed.
- **The timing comparisons are numpy timings on one machine.** They establish the ranking and
  the reason for it, and they are explicitly *not* offered as flop counts. Where the measured
  ratio disagrees with the flop count, the audit says so and explains it by the memory
  hierarchy rather than adjusting either number.
- **`gcv_score` beat the L-curve on one problem family**, a discretised first kind integral
  equation, over 20 random instances. That is enough to report a 16 to 4 win and a median
  ratio of 1.25 against 3.90, and it is not enough to call GCV better in general.
- **The GPS results use a synthetic constellation and a spherical Earth.** They are correct as
  a demonstration of the method and of dilution of precision, and they are not a model of the
  real system, which has an ellipsoidal Earth, relativistic corrections, atmospheric delay and
  ephemeris error.
- **Solution 33.5.2's analysis of errors-in-variables GPS is reasoned, not measured.** The
  order-of-magnitude argument is given and the conclusion is stated as an expectation.
