# Course Architecture

This is the design record for the repository. It explains what the course contains, why it
is ordered the way it is, and how it will be checked.

For the plain list of all 98 lessons, see [COURSE_MAP.md](COURSE_MAP.md).

---

## 1. Design principle

The two source books are used as knowledge sources and as hidden completeness checklists.
They are **not** the shape of the course.

The course is organised by numerical analysis concepts and by what a learner needs to know
before they can understand the next thing. A learner never has to know which book a topic
came from, and never has to jump between books.

Where the two books overlap, their treatments are merged into one stronger lesson. Where a
book has something the other does not, that material is kept and placed where it fits best
in the concept order. Where a graduate course needs something neither book has, it is
written and clearly marked as supplementary.

The course runs from the very basic (what a binary number is) to genuinely advanced
(Krylov eigensolvers, multigrid, randomized SVD, automatic differentiation).

---

## 2. Shape of the course

14 parts, 98 lessons.

| Part | Name | Lessons | Count |
|---|---|---|---|
| 1 | Foundations of Numerical Computing | 01 to 08 | 8 |
| 2 | Nonlinear Equations and Root Finding | 09 to 14 | 6 |
| 3 | Direct Methods for Linear Systems | 15 to 22 | **8** |
| 4 | Iterative and Krylov Subspace Methods | 23 to 28 | 6 |
| 5 | Orthogonality, QR and Least Squares | 29 to 34 | 6 |
| 6 | Eigenvalue Problems and the SVD | 35 to 43 | 9 |
| 7 | Interpolation | 44 to 53 | 10 |
| 8 | Approximation Theory and Transforms | 54 to 60 | 7 |
| 9 | Numerical Differentiation and Integration | 61 to 66 | 6 |
| 10 | Ordinary Differential Equations | 67 to 76 | 10 |
| 11 | Partial Differential Equations | 77 to 83 | 7 |
| 12 | Numerical Optimization | 84 to 90 | 7 |
| 13 | Stochastic and Monte Carlo Methods | 91 to 93 | 3 |
| 14 | Numerical Analysis in Machine Learning and AI | 94 to 98 | 5 |

Numerical linear algebra (parts 3 to 6) is 29 of the 98 lessons, the largest block by a
wide margin. That is deliberate.

---

## 3. Learning order and why

```
Part 1  Foundations
        floating point -> error -> conditioning -> stability -> cost
              |
              v
Part 2  Root finding in one variable
        the cheapest setting for convergence order, sensitivity, backward error
              |
              v
Part 3  Direct linear systems
        norms -> elimination -> LU -> pivoting -> conditioning -> Cholesky -> sparse
              |
              v
Part 4  Iterative and Krylov solvers
        splitting -> CG -> preconditioning -> Krylov/Arnoldi/Lanczos -> GMRES -> multigrid
              |
              v
Part 5  Orthogonality, QR, least squares
        projection -> Gram-Schmidt -> Householder/Givens -> rank -> nonlinear least squares
              |
              v
Part 6  Eigenvalues and SVD
        theory -> power methods -> QR algorithm -> symmetric -> Krylov -> SVD
              |
              +--------------------------+
              v                          v
Part 7  Interpolation           Part 8  Approximation and transforms
              |                          |
              +------------+-------------+
                           v
Part 9  Differentiation and integration
                           |
                           v
Part 10 ODEs  ------->  Part 11 PDEs
                           |
                           v
Part 12 Optimization  ->  Part 13 Stochastic methods
                           |
                           v
Part 14 Machine learning and AI connections
```

Four ordering choices are worth explaining, because a reader might expect otherwise:

**Root finding before linear systems.** Solving `f(x) = 0` in one variable is the cheapest
possible place to make forward error, backward error, sensitivity and order of convergence
concrete. Once those ideas are solid in one dimension, the same ideas in matrix form take
far less work.

**Iterative solvers before QR.** GMRES minimises a residual over a Krylov space, which is a
least squares problem, so it looks like least squares should come first. But at that point
the learner only needs "make the residual as small as possible", and the QR machinery that
actually implements it is developed immediately afterwards in Part 5 and referred back to.
The other ordering forces orthogonalization on the learner before they have seen why anyone
wants it. Both lessons flag the link explicitly.

**Interpolation after linear algebra.** Interpolation is usually taught early because it
feels elementary. But the honest version of it needs the conditioning of the Vandermonde
system, and splines need a tridiagonal solve. Putting it after Part 3 means those can be
used rather than hand-waved.

**Optimization after ODEs.** This lets Part 12 reuse Newton's method, line searches built
on root finding, and the conjugate gradient idea from Part 4, and it sets up Part 14 where
gradient descent is read as an ODE discretization.

---

## 4. How the Trefethen and Bau depth requirement is handled

Trefethen and Bau, *Numerical Linear Algebra*, drives the depth of the linear algebra
material. It was supplied as a 40-lecture concept map, `TrefethenBau.md`, after the initial
design, and the course was audited against it. That audit is
[`_planning/TrefethenBau_Course_Mapping.md`](_planning/TrefethenBau_Course_Mapping.md) and it
resolves all 40 lectures with no unaccounted items.

The audit produced exactly one structural change, recorded in
[`_planning/CURRICULUM_CHANGE_LOG.md`](_planning/CURRICULUM_CHANGE_LOG.md): a new lesson 16,
*Orthogonality and Projectors*, covering the two lectures that had no home in the original
plan. Everything else was absorbed by strengthening lessons that already existed.

**One. Linear algebra gets the deepest treatment in the course anyway.** Parts 3 to 6 are
29 lessons, including 9 lessons on eigenvalue problems and the SVD alone.

**Two. Sauer independently supports most of the requested depth.** This was confirmed by
reading the actual pages of the scanned book, not the table of contents, so most Trefethen and
Bau material has a second source behind it as well:

| Requested topic | Actually present in Sauer | Where verified |
|---|---|---|
| Conditioning, error magnification | Yes, Definitions 2.3 and 2.4 | book p89 |
| Forward and backward error | Yes | book p89, and section 1.3 |
| Classical and modified Gram-Schmidt | Yes, with the loss-of-orthogonality experiment | book p228 |
| Householder reflectors | Yes, section 4.3.3 | book p228 |
| Krylov subspaces, GMRES | Yes, with orthogonalization built in | book p235 |
| Conjugate gradient | Yes, A-inner product and conjugacy | book p128 |
| Preconditioning | Yes, section 2.6.4 | table of contents plus section |
| Power, inverse power, Rayleigh quotient iteration | Yes, section 12.1 | table of contents |
| Simultaneous iteration and the QR algorithm | Yes, derived in full | book p566 |
| Hessenberg form, real Schur form | Yes, sections 12.2.2 to 12.2.3 | table of contents |
| SVD as unit sphere to hyperellipse | Yes, Figures 12.2 and 12.3 | book p579 |
| Computing the SVD, condition squaring | Yes, section 12.4.4 | book p591 |

Gupta adds Givens rotations, Jacobi rotations, Householder tridiagonalization, Sturm
sequences for symmetric tridiagonal matrices, Gerschgorin and Brauer bounds, and the
Rutishauser LR method.

**Three. Real gaps are filled and labelled.** A graduate numerical linear algebra course
needs some things no supplied source has. Those lessons carry a **Supplementary / Advanced
Material** banner naming exactly what is not from a supplied source. The full list lives in
[`_planning/coverage_data.py`](_planning/coverage_data.py) under source code `SUP` (68 items).
Items that are in Trefethen and Bau are attributed to it under source code `S3` (75 items) and
are **not** labelled supplementary. The remaining supplementary linear algebra items are:

| Supplementary topic | Lesson |
|---|---|
| General norm theory, induced norms, norm equivalence | 15 |
| Growth factor and the theory behind pivoting | 17 |
| Condition estimation without forming the inverse, iterative refinement | 21 |
| Incomplete Cholesky preconditioner | 24 |
| Arnoldi and Lanczos as standalone algorithms, Ritz values | 25 |
| Restarted GMRES, MINRES, BiCGSTAB | 26 |
| Multigrid: smoothing, coarse grid correction, the V-cycle | 27 |
| Pseudoinverse, Tikhonov regularization, the L-curve | 31 |
| QR with column pivoting, numerical rank, total least squares | 32 |
| Bauer-Fike theorem, Schur decomposition existence | 34 |
| Divide and conquer for the symmetric tridiagonal eigenproblem | 37 |
| Krylov eigensolvers with implicit restarting | 38 |
| Generalized eigenvalue problem, Cholesky reduction, QZ | 39 |
| Golub-Kahan bidiagonalization, one-sided Jacobi SVD | 41 |
| Eckart-Young-Mirsky theorem, randomized SVD | 42 |

**Four. Everything important is written from scratch**, then checked against NumPy or SciPy
*and* against the defining identity, not just printed to the screen.

---

## 5. Repository layout

```
numerical-analysis/
├── README.md                          front page, notation, how to run
├── COURSE_ARCHITECTURE.md             this file
├── COURSE_MAP.md                      all 98 lessons, generated
├── LEARNING_PATH.md                   suggested routes through the course
├── COURSE_DEPENDENCIES.md             the dependency graph
├── requirements.txt
│
├── 01_foundations/                    lessons 01-08
├── 02_root_finding/                   lessons 09-14
├── 03_direct_linear_systems/          lessons 15-22
├── 04_iterative_and_krylov/           lessons 23-28
├── 05_least_squares_and_qr/           lessons 29-34
├── 06_eigenvalues_and_svd/            lessons 35-43
├── 07_interpolation/                  lessons 44-53
├── 08_approximation_and_transforms/   lessons 54-60
├── 09_differentiation_and_integration/lessons 61-66
├── 10_ordinary_differential_equations/lessons 67-76
├── 11_partial_differential_equations/ lessons 77-83
├── 12_optimization/                   lessons 84-90
├── 13_stochastic_methods/             lessons 91-93
├── 14_ml_and_ai_connections/          lessons 94-98
│
├── src/nalib/                         the reusable from-scratch library
├── tests/                             pytest suite for src/nalib
├── solutions/                         worked solutions to exercises
├── datasets/                          small generated datasets
├── figures/                           figures saved by notebooks
├── verification/                      execution, test, numerical and audit reports
└── _planning/                         internal QA artifacts, not for learners
```

Every lesson is a pair with the same name: `NN_topic.ipynb` and `NN_topic.md`. The notebook
is the runnable lesson, the markdown is the same lesson to read without running anything.
They state the same mathematics and reach the same conclusions.

`_planning/` holds the source inventory, the lesson map, the coverage data and the scripts
that generate `COURSE_MAP.md` and the coverage checklist. Learners never need to open it.

---

## 6. The `src/nalib` library

Notebooks teach algorithms by building them in front of the reader. But an algorithm built
in lesson 16 is needed again in lesson 50, and copy-pasting it 15 times would be a
maintenance and correctness disaster.

So each algorithm is written twice on purpose:

1. **In the notebook**, spelled out step by step with the mathematics next to it. This is
   the teaching version.
2. **In `src/nalib`**, as a clean, documented, tested function. This is the version later
   lessons import, and the version the test suite exercises.

The notebook version and the library version are checked against each other numerically in
the lesson where the algorithm is introduced. Planned modules:

| Module | Contents |
|---|---|
| `nalib/floatingpoint.py` | machine epsilon tools, `fl` simulation, compensated summation |
| `nalib/roots.py` | bisection, regula falsi, fixed point, Newton, secant, Muller, Brent |
| `nalib/polyroots.py` | Horner, Birge-Vieta, deflation, Sturm, Bairstow, Graeffe |
| `nalib/linalg_direct.py` | LU, PA=LU, Cholesky, LDL, Thomas, substitution routines |
| `nalib/norms.py` | vector and matrix norms, condition numbers, condition estimation |
| `nalib/iterative.py` | Jacobi, Gauss-Seidel, SOR, CG, preconditioners, multigrid |
| `nalib/krylov.py` | Arnoldi, Lanczos, GMRES, MINRES |
| `nalib/qr.py` | classical and modified Gram-Schmidt, Householder, Givens, column pivoting |
| `nalib/leastsq.py` | normal equations, QR and SVD solvers, Gauss-Newton, Levenberg-Marquardt |
| `nalib/eigen.py` | power methods, QR algorithm, Hessenberg, Jacobi, tridiagonalization |
| `nalib/svd.py` | bidiagonalization, one-sided Jacobi, truncated and randomized SVD |
| `nalib/interpolate.py` | Lagrange, Newton, divided differences, Hermite, splines, Bezier |
| `nalib/approx.py` | orthogonal polynomials, Chebyshev, Pade, DFT and FFT, DCT |
| `nalib/quadrature.py` | Newton-Cotes, Romberg, adaptive, Gauss rules |
| `nalib/ode.py` | Euler, RK, embedded pairs, multistep, implicit solvers, BVP solvers |
| `nalib/pde.py` | finite difference stencils, parabolic, hyperbolic, elliptic solvers, ADI |
| `nalib/optimize.py` | golden section, Nelder-Mead, gradient methods, BFGS, trust region |
| `nalib/stochastic.py` | LCG, Box-Muller, Monte Carlo, Halton, Euler-Maruyama, Milstein |
| `nalib/autodiff.py` | forward mode dual numbers, reverse mode tape |

---

## 7. Standards every lesson must meet

1. Mathematics stated precisely: definitions, assumptions, theorems, derivations, and proof
   sketches where a full proof would be too long.
2. Every important algorithm written from scratch in NumPy, never only a library call.
3. Every from-scratch implementation compared against SciPy or NumPy **and** against the
   defining identity, for example `norm(A - Q @ R)` and `norm(Q.T @ Q - I)` for QR.
4. Conditioning, stability, error analysis and complexity are their own sections, not
   afterthoughts.
5. At least one numerical experiment that makes a phenomenon visible.
6. Every plot has a title, labelled axes, a legend when there is more than one series, and
   a sentence saying what the reader should take away.
7. Exercises at five levels: conceptual, mathematical, computational, experimental, and
   advanced or open-ended.
8. Reproducible. `rng = np.random.default_rng(SEED)` wherever randomness appears.
9. No placeholders, no bare `pass` bodies, no invented output.
10. The `.md` and the `.ipynb` state the same mathematics and the same conclusions.

A lesson adapts these sections to its topic. A theory-heavy lesson will not force a
complexity section it does not need, and an algorithm-heavy lesson will not pad out a
history section.

### 7.1 Every library routine must be independent of its input size

This is a hard rule on everything in `src/nalib`, and it binds every part of the course.

A function that accepts a matrix must work for **1x1**, for **2x3**, for **34x65**, and for
anything else. A function that accepts a square matrix must work from **1x1 upwards to any
size**. Nothing may quietly assume "at least 2", "square", "even", "a power of two", or "big
enough for the interesting case".

The lessons demonstrate specific examples, and that is expected. The **library** must not
inherit those examples' shapes.

| Requirement | What it means in practice |
|---|---|
| No hidden minimum size | Loops and slices must be correct when they run zero times. `v[1:]` on a length-1 vector is empty, not an error to avoid by assuming $n \ge 2$ |
| Both orientations | A routine taking an $m \times n$ matrix must handle $m < n$, $m = n$ and $m > n$ |
| Every subspace dimension | A projector onto a $k$-dimensional subspace must work for $k = 1$, for $k = n$, and everything between |
| Shapes come from the input | Never from a constant. A Jacobian of $F : \mathbb{R}^n \to \mathbb{R}^m$ is $m \times n$, with $m$ and $n$ unrelated |
| Genuine restrictions are allowed, silence is not | Bairstow needs degree $\ge 2$ because it finds a **quadratic** factor. That is mathematics, and it raises a clear `ValueError` saying so. It does not return a wrong answer |

**This is enforced permanently**, not audited once, by
[`tests/test_dimension_independence.py`](tests/test_dimension_independence.py). That module
sweeps every size-varying routine across degenerate sizes (1x1, length-1 vectors, rank-one
subspaces), the small sizes where off-by-one errors live (2, 3, 4), both orientations of
rectangular shapes, and awkward non-round sizes (7, 13, 23, 34).

Each check tests the routine's **defining identity** at each size, not merely that it returns
without raising. **A wrong answer at $n = 1$ is a worse failure than a crash**, because it is
silent.

### 7.2 The same rule applies to values, and to notebook code

**Values, not only sizes.** A routine must be correct across the whole valid range of its
inputs: negative, zero, decimal, very large, very small, subnormal, and mixed magnitudes. Code
that works only for inputs near 1 is as restricted as code that works only at $n = 10$. The
sweep therefore also runs every routine across magnitudes from $10^{-300}$ to $10^{300}$.

Two defects this caught, both silent wrong answers rather than crashes:

| Defect | Effect |
|---|---|
| Bracketing methods tested for a sign change with `f(a)*f(b) < 0` | The product **underflows to exactly 0** below about $10^{-154}$ and **overflows to inf** above $10^{154}$, and `inf * 0` is `nan`. Measured: bisection on $f(x) = 10^{-200}(x - 1.3)$ over $[0,3]$ returned **3.0 instead of 1.3**. Fixed by comparing signs, which does no arithmetic |
| The stable quadratic formula computed `b*b - 4*a*c` directly | With all coefficients near $10^{-300}$, `b*b` **underflows to 0**, so the discriminant comes out 0 and a complex pair is reported as a double root. Fixed by scaling by a **power of two** first, which is exact in binary and so introduces no rounding of its own |

**Notebook code too.** The functions defined inside lessons are implementations, not scratch
work, and the same rule binds them. The demonstration *data* may be specific, and should be,
but the *algorithm* must not be. Concretely: no reliance on module-level globals, no assumed
input magnitude, and no logic that only works for the example being shown.

The `.py` library and the `.ipynb` teaching version of an algorithm must implement the **same
general logic**. Where a lesson shows a simpler variant to make a point, it must say so and
measure the difference rather than leaving the reader with the weaker version.

**What the scanner reads.** `_planning/check_generality.py` scans six sets of files, because the
source is where a fix is made but the notebook is what a reader actually opens and runs:

| Scanned | Why |
|---|---|
| `_planning/lessons_src/*.md` | the authored source, where a fix belongs |
| `[0-9][0-9]_*/*.ipynb` | the built notebooks, parsed as JSON so only `code` cells are read |
| `[0-9][0-9]_*/*.md` | the built markdown lessons |
| `src/nalib/*.py` | the library |
| `tests/*.py` | test code a reader may copy |
| `solutions/*.md` | worked solutions |

Test files are exempt from the *size literal* patterns, because choosing a size is what a test
case is for, and `tests/test_dimension_independence.py` is the tool that sweeps the range
deliberately. They are **not** exempt from the fixed-index pattern, since `s[2]` where `s[-1]`
was meant is a bug wherever it appears.

---

## 8. Notation, fixed across the whole course

| Symbol | Meaning |
|---|---|
| `x`, `y`, `b` | column vectors, lower case |
| `A`, `Q`, `R`, `L`, `U` | matrices, upper case |
| `x_hat` (written as x with a hat) | a computed, inexact quantity |
| `n` | dimension of a square system, or number of unknowns |
| `m` | number of rows in an overdetermined system, so `A` is m by n with m >= n |
| `k` | iteration index |
| `h` | step size, in space or in time |
| `eps_mach` | machine epsilon, 2 to the power -52 in IEEE double |
| `u` | unit roundoff, half of machine epsilon |
| `kappa(A)` | condition number of `A` in the stated norm |
| `rho(A)` | spectral radius of `A` |
| `sigma_i` | the i-th singular value, ordered largest first |
| `lambda_i` | the i-th eigenvalue |
| `r = b - A x_hat` | residual |
| `O(n^3)` | asymptotic operation count |

Errors are always named exactly: forward error, backward error, absolute error, relative
error. They are never used interchangeably.

---

## 9. Verification strategy

Every phase ends with the same loop, and a phase is not finished until it passes.

```
write lesson  ->  execute notebook  ->  read the traceback if it fails
      ^                                          |
      |                                          v
   fix code   <-  check the numbers  <-  re-execute until clean
```

Concretely:

- **Execution.** Every notebook is run end to end with `jupyter nbconvert --execute`.
  Pass or fail, wall time, warnings and tracebacks all go into
  `verification/notebook_execution_report.md`.
- **Self-checking notebooks.** Notebooks assert their own numerical claims, so a wrong
  number becomes an execution failure instead of something a reader has to spot.
- **Identity checks.** `A - LU`, `PA - LU`, `A - QR`, `Q-transpose Q - I`,
  `A - U S V-transpose`, `A v - lambda v`, `b - A x`, and orthogonality of the least squares
  residual. Tolerances come from machine epsilon and the condition number. They are never
  loosened to make something pass.
- **Library cross-check.** From-scratch results are compared against NumPy and SciPy. Any
  disagreement is investigated and fixed, never hidden.
- **Tests.** `pytest tests/` covers `src/nalib` with normal cases, edge cases, known
  analytic cases, seeded random cases, ill-conditioned cases, and singular or near-singular
  cases.
- **Coverage audit.** Every one of the 507 tracked concepts in
  `_planning/INTERNAL_COVERAGE_CHECKLIST.md` is ticked off against the lesson that covers
  it, with separate flags for covered, implemented, experimented, tested and verified.
- **Placeholder sweep.** The whole repository is searched for TODO, FIXME, TBD, bare `pass`
  and similar before the final audit.

Deliberate failures are allowed only when the failure is the point of the experiment. In
that case the notebook says so before it runs, catches the failure, and explains it. Every
such case is listed in the execution report so nobody mistakes it for a bug.

---

## 10. Phase plan

| Phase | Content | Lessons | Deliverable |
|---|---|---|---|
| 1 | Discovery and design | none | this file, source inventory, coverage checklist, course map |
| 2 | Foundations | 01-08 | 8 lessons, `nalib` core, tests, execution report |
| 3 | Root finding and direct linear systems | 09-22 | 14 lessons |
| 4 | Iterative, Krylov, QR, least squares, eigenvalues, SVD | 23-43 | 21 lessons, the deepest block |
| 5 | Interpolation, approximation, transforms, calculus | 44-66 | 23 lessons |
| 6 | ODEs and PDEs | 67-83 | 17 lessons |
| 7 | Optimization, stochastic methods, ML and AI | 84-98 | 15 lessons |
| 8 | Full repository verification | none | five verification reports and the final audit |

Each phase runs the full generate, execute, test, verify, fix, re-execute loop before the
next one starts.
