# Audit: Part 3, Direct Methods for Linear Systems

A part-level audit. [FINAL_AUDIT.md](FINAL_AUDIT.md) carries the current overall position and
links to every part-level audit, so that list is maintained in one place rather than repeated
here and going stale each time a part lands.

Everything below is measured from the repository by [`run_all.py`](run_all.py). No number in
this document was typed in by hand.

---

## Verdict

**Part 3 passes.** Eight lessons, all executed, all tested, all numerically verified, no
placeholders, no unpaired files, no hardcoded dimensions.

| Check | Result |
|---|---|
| Notebooks executed end to end | **8 of 8 pass** (22 of 22 across the repository) |
| Automated tests | **1166 of 1166 pass** (79 new in Part 3) |
| Assertions inside the Part 3 lessons | **101, all passing** |
| Concepts covered from the sources | **57 new, 147 of 507 total** |
| Concepts outstanding in Parts 1 to 3 | **0** |
| Worked exercise solutions | **136**, one for every exercise |
| Generality scan | **0 candidates** |

---

## A. Curriculum completeness

Part 3 was planned as 8 lessons. Every one exists, including lesson 16, the single structural
addition the Trefethen and Bau audit produced.

| Lesson | Title | Figures | Assertions |
|---|---|---:|---:|
| 15 | Vectors, Matrices and Norms | 3 | 15 |
| 16 | Orthogonality and Projectors | 3 | 21 |
| 17 | Gaussian Elimination and LU | 1 | 13 |
| 18 | Pivoting and PA = LU | 1 | 12 |
| 19 | Conditioning of Linear Systems | 2 | 12 |
| 20 | Symmetric Positive Definite and Cholesky | 1 | 14 |
| 21 | Banded, Sparse and Structured Systems | 1 | 8 |
| 22 | Condition Estimation and Iterative Refinement | 1 | 6 |

**Nothing planned was dropped.**

## B. Source completeness

| Source | Chapter | Concepts | Status |
|---|---|---:|---|
| Sauer | 2, Systems of Equations | 24 of 24 | **complete** |
| Trefethen and Bau | L1, L2, L3, L6, L17, L20, L21, L22 | 24 of 24 | **complete** |
| Gupta | 5, Systems of Linear Equations | 12 of 12 | **complete** |
| Supplementary | LDL, sparse formats, condition estimation, refinement | 4 | **complete** |

**Lesson 16 is the one new lesson** the Trefethen audit produced, covering lectures 2 and 6
which had no home in the original plan. Its justification is recorded in
[`_planning/TrefethenBau_Course_Mapping.md`](../_planning/TrefethenBau_Course_Mapping.md)
section 8, and the change in
[`_planning/CURRICULUM_CHANGE_LOG.md`](../_planning/CURRICULUM_CHANGE_LOG.md).

## C. Numerical linear algebra depth

**This is the part where the Trefethen and Bau requirement is first cashed in**, and the
material it demanded is present and measured rather than mentioned:

| Requested depth | Where, and what is measured |
|---|---|
| The column view of $Ax$ and $AB$ | Lesson 15 section 1 and 2, with the outer product view shown building rank one piece at a time |
| Norm-based error bounds as a working tool | Lesson 15 section 8, verified across $\kappa$ from $10^0$ to $10^9$ |
| Orthogonality as the foundation of stability | Lesson 16, with the 400-trial experiment showing orthogonal steps preserve the error exactly while general ones span $5\times10^5$ |
| Projectors, oblique against orthogonal | Lesson 16 section 7, with $\|P\|_2 = 1/\cos\theta_{\max}$ verified to $10^{-8}$ over six decades |
| Componentwise stability of back substitution | Lesson 17 section 7, componentwise backward error measured below $nu$ at every size |
| Growth factor theory | Lesson 18, Wilkinson's matrix attaining $2^{n-1}$ with ratio 1.000000 at every size |
| Worst case against practice | Lesson 18 sections 5 and 6, median growth fitted at $n^{0.54}$ against a bound of $1.7\times10^{38}$ |
| Perturbation theory for $Ax=b$ | Lesson 19, both theorems **derived** rather than quoted |

## D. Computational completeness

Eight library modules added, 95 public functions, 2305 lines:

| Module | Contents |
|---|---|
| `linalg` | vector and matrix norms, the three views of a product, norm axioms, equivalence constants |
| `orthogonality` | orthogonal matrices, Householder reflectors, projectors, principal angles |
| `lu` | elimination, LU, Doolittle, Crout, Gauss-Jordan, Cramer, triangular solves, flop counts |
| `pivoting` | partial and complete pivoting, $PA = LU$, growth factor, Wilkinson's matrix |
| `linsys` | residuals, error bounds, diagnosis, Hilbert, Vandermonde, beam matrices |
| `cholesky` | Cholesky, LDL, inertia, positive definiteness testing |
| `banded` | Thomas, banded LU, COO and CSR, fill-in, reverse Cuthill-McKee |
| `refinement` | Hager condition estimation, exact residuals, iterative refinement |

**Every algorithm is checked against a trusted reference and against its defining identity:**

| From scratch | Checked against |
|---|---|
| `cholesky` | `numpy.linalg.cholesky`, and $\|A - LL^T\|$ |
| `plu_factor` | $\|PA - LU\|$ and `numpy.linalg.solve` |
| `thomas` | a dense solve, at sizes from 1 to 200 |
| `matrix_norm` | `numpy.linalg.norm`, all four norms, all shapes |
| `condition_estimate` | the exact $\kappa_1$, ratio measured 0.70 to 1.000006 |
| `second_difference_eigenvalues` | `eigvalsh`, exact to $10^{-10}$ |
| `exact_solution` | its own residual, below $10\,u$ |

## E. Defects this process caught

Every one is a real bug or a wrong claim found by running things.

| What | How it was caught |
|---|---|
| **All four bracketing methods used `f(a)*f(b) < 0`.** The product underflows to exactly 0 below $10^{-154}$: bisection on $10^{-200}(x-1.3)$ returned **3.0 instead of 1.3** | Value-range sweep across 600 orders of magnitude. Fixed with a shared sign helper; all four now hold to $5\times10^{-15}$ throughout |
| **Iterative refinement appeared not to work.** `math.fsum` fixes the summation but the **products** are rounded first | Comparing three residual precisions. With an exact residual the error falls to **exactly zero** in three steps; with `fsum` or working precision it never improves |
| **The measurement trap.** $\mathbf{b} = A\mathbf{x}_{\text{true}}$ is rounded, so the exact answer to the **stored** system differs from $\mathbf{x}_{\text{true}}$ by $\kappa u$ | Measuring against `exact_solution` instead. The floor was $1.1\times10^{-6}$ at $\kappa = 10^{12}$, exactly where refinement seemed to stall |
| **`inertia` returned an answer for a nonsymmetric matrix**, silently reading one triangle | A sweep over matrix kinds. Now rejects, because inertia is defined only for symmetric matrices |
| **Growth factor can be below 1**, contradicting a test I wrote | About 6 percent of random 8 by 8 matrices, smallest 0.688. The definitional subtlety is now recorded in the module |
| **Claimed CSR beats dense by 3000 times at $n = 10^4$.** The real figure is **1429** | A test with the arithmetic done properly |
| **Claimed random growth fits $n^{2/3}$**, the folklore figure | Fitting the measurement: the median fits $n^{0.54}$. The lesson reports the fit and flags the conjecture as a conjecture |
| **Hager's estimator reported as "always an underestimate"** | It exceeds the reference by up to $1.000006$, which is the **reference's** own error from `inv(A)`, not the estimator's. Stated precisely instead |
| Hardcoded dimensions across six lessons: `range(3)`, `np.eye(25)` paired with `== 5.0`, `s[2]`, `range(60)` | The generality scanner, which now also reads the built `.ipynb` files |

## Statistics for Part 3

```text
Lessons written                              : 8   (15 to 22)
Words in the 8 markdown lessons              : 45313
Figures generated                            : 13
From-scratch functions inside notebooks      : 10
Library modules added                        : 8
Public library functions added               : 95
Lines of library code added                  : 2305
Automated tests added                        : 79
Assertions inside the Part 3 notebooks       : 101
Notebooks passed                             : 8 of 8
Concepts covered by Part 3                   : 57
Source chapters fully absorbed               : 3
Worked exercise solutions written            : 136
Concepts outstanding in Part 3               : 0
```

Repository totals after Part 3:

```text
Lessons written                              : 22 of 96
Concepts covered                             : 147 of 507  (29.0 percent)
Library modules                              : 17
Public library functions                     : 205
Lines of library code                        : 5216
Test modules                                 : 14
Automated tests                              : 1166
Lines of test code                           : 5062
Figures generated                            : 39
Notebooks passed                             : 22 of 22
Placeholders found                           : 0
Hardcoded dimensions found                   : 0
```

## What is deliberately not claimed

- This is not the final audit. 74 of 96 lessons remain.
- The **timing** comparisons in lessons 17, 20 and 21 compare a Python loop against LAPACK, and
  each says so explicitly. The flop counts are the honest algorithmic comparison and the lessons
  point at them.
- The growth factor result for random matrices is **empirical**. That practical matrices have
  small growth has no proof, and lesson 18 says so at length rather than implying otherwise.
- The exact-residual refinement uses rational arithmetic, which is slow. Production code uses
  double-double; the mechanism is identical and the lesson states the difference.
- Lesson 19's beam trade-off uses a representative $1/n^2$ discretisation error rather than a
  measured one, and the column is labelled as a model.
