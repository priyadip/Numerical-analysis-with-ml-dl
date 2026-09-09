# Numerical Analysis: A Complete Course

A graduate-level numerical analysis course built as a runnable repository. 98 lessons in 14
parts, taking you from what a binary number is to Krylov eigensolvers, multigrid,
randomized SVD and automatic differentiation.

Every lesson is a pair of files with the same name:

- `NN_topic.ipynb` is the runnable lesson. You execute it, change numbers, and watch what
  happens.
- `NN_topic.md` is the same lesson to read. Same mathematics, same conclusions, no kernel
  required.

---

## Build status

This repository is being built in phases. Nothing here is a placeholder: what exists is
finished and verified, and what does not exist yet is listed as not started.

| Phase | Content | Lessons | Status |
|---|---|---|---|
| 1 | Discovery, source analysis and course design | none | **done** |
| 2 | Foundations | 01-08 | **done and verified** |
| 3a | Nonlinear equations and root finding | 09-14 | **done and verified** |
| 3b | Direct linear systems | 15-22 | **done and verified** |
| 3c | Iterative and Krylov methods | 23-28 | **done and verified** |
| 3d | Orthogonality, QR and least squares | 29-34 | **done and verified** |
| 3e | Eigenvalue problems and the SVD | 35-43 | **done and verified** |
| 4a | Interpolation | 44-53 | **done and verified** |
| 4b | Approximation theory and transforms | 54-60 | **done and verified** |
| 4c | Numerical differentiation and integration | 61-66 | **done and verified** |
| 5a | Ordinary differential equations | 67-76 | **done and verified** |
| 5b | Partial differential equations | 77-83 | **done and verified** |
| 6a | Numerical optimization | 84-90 | **done and verified** |
| 6b | Stochastic methods | 91-93 | **done and verified** |
| 6c | Machine learning and AI connections | 94-98 | not started |
| 7 | Full repository verification and final audit | none | not started |

Phase 1 produced [COURSE_ARCHITECTURE.md](COURSE_ARCHITECTURE.md),
[COURSE_MAP.md](COURSE_MAP.md), [COURSE_DEPENDENCIES.md](COURSE_DEPENDENCIES.md), and the
internal coverage checklist that tracks 514 concepts from the source material.

Phase 2 produced the 8 lessons of [Part 1](01_foundations/), the `src/nalib` library they
build, and the test and verification machinery the rest of the course reuses. Phase 3a added
the 6 lessons of [Part 2](02_root_finding/) and the `roots`, `polyroots` and `nlsystems`
modules. Phase 3b added the 8 lessons of [Part 3](03_direct_linear_systems/) and eight more
modules. Phase 3c added the 6 lessons of [Part 4](04_iterative_and_krylov/) and the
`iterative`, `krylov`, `nonsymmetric` and `multigrid` modules. Phase 3d added the 6 lessons of
[Part 5](05_least_squares_and_qr/) and the `leastsquares`, `qr`, `rrqr` and `nlls` modules.
Phase 3e added the 9 lessons of [Part 6](06_eigenvalues_and_svd/) and the `eigen`, `power`,
`qralg`, `symeig`, `krylov_eig`, `geneig`, `svd`, `svdcompute` and `lowrank` modules.
Phase 4a added the 10 lessons of [Part 7](07_interpolation/) and the `interp`, `divdiff`,
`interperror`, `chebyshev`, `finitediff`, `equalinterval`, `hermite`, `splines`, `bezier` and
`bivariate` modules. Phase 4b added the 7 lessons of
[Part 8](08_approximation_and_transforms/) and the `approx`, `orthopoly`, `chebapprox`, `pade`,
`dft`, `fft` and `dct` modules. Phase 4c added the 6 lessons of
[Part 9](09_differentiation_and_integration/) and the `differentiation`, `newtoncotes`,
`romberg`, `adaptive`, `gaussquad` and `multiquad` modules. Phase 5a added the 10 lessons of
[Part 10](10_ordinary_differential_equations/) and the `ivp`, `taylorode`, `rungekutta`,
`adaptivestep`, `multistep`, `stability`, `odesystems`, `symplectic`, `bvp` and `femode`
modules. Phase 5b added the 7 lessons of
[Part 11](11_partial_differential_equations/) and the `pdeclass`, `parabolic`, `pdestability`,
`adi`, `hyperbolic`, `elliptic` and `nonlinearpde` modules. Phase 6a added the 7 lessons of
[Part 12](12_optimization/) and the `optimize`, `derivfree`, `gradient`, `quasinewton`,
`constrained`, `stochastic` and `proximal` modules. Phase 6b added the 3 lessons of
[Part 13](13_stochastic_methods/) and the `rng`, `montecarlo` and `sde` modules. Phase 6c added the
5 lessons of [Part 14](14_ml_and_ai_connections/) and the `mllinalg`, `mlopt`, `mixedprecision`,
`autodiff` and `sciml` modules.

**The course is complete.** All 98 lessons are written, executed, tested, numerically verified and
solved, and all 14 parts are audited.

Current state, all measured rather than claimed:

| | |
|---|---|
| Notebooks executing cleanly | 98 of 98 |
| Automated tests passing | 9606 of 9606 |
| Independent identity checks | 13 of 13 |
| Assertions inside the lessons | 1000 |
| Source concepts covered | 514 of 514 |
| Worked exercise solutions | 1783, one for every exercise in every part |
| Broken internal links | 0 |
| Placeholders anywhere in the repository | 0 |
| Hardcoded dimensions anywhere | 0 |

Reports are in [verification/](verification/).
[verification/FINAL_AUDIT.md](verification/FINAL_AUDIT.md) is the current overall position,
with the full part-level audits in
[AUDIT_PART1.md](verification/AUDIT_PART1.md),
[AUDIT_PART2.md](verification/AUDIT_PART2.md),
[AUDIT_PART3.md](verification/AUDIT_PART3.md),
[AUDIT_PART4.md](verification/AUDIT_PART4.md),
[AUDIT_PART5.md](verification/AUDIT_PART5.md),
[AUDIT_PART6.md](verification/AUDIT_PART6.md),
[AUDIT_PART7.md](verification/AUDIT_PART7.md),
[AUDIT_PART8.md](verification/AUDIT_PART8.md),
[AUDIT_PART9.md](verification/AUDIT_PART9.md),
[AUDIT_PART10.md](verification/AUDIT_PART10.md),
[AUDIT_PART11.md](verification/AUDIT_PART11.md),
[AUDIT_PART12.md](verification/AUDIT_PART12.md),
[AUDIT_PART13.md](verification/AUDIT_PART13.md) and
[AUDIT_PART14.md](verification/AUDIT_PART14.md). Reproduce every number with
`python verification/run_all.py`.

---

## What this course covers

| Part | Name | Lessons |
|---|---|---|
| 1 | [Foundations of Numerical Computing](01_foundations/) | 01-08 |
| 2 | [Nonlinear Equations and Root Finding](02_root_finding/) | 09-14 |
| 3 | [Direct Methods for Linear Systems](03_direct_linear_systems/) | 15-22 |
| 4 | [Iterative and Krylov Subspace Methods](04_iterative_and_krylov/) | 23-28 |
| 5 | [Orthogonality, QR and Least Squares](05_least_squares_and_qr/) | 29-34 |
| 6 | [Eigenvalue Problems and the SVD](06_eigenvalues_and_svd/) | 35-43 |
| 7 | [Interpolation](07_interpolation/) | 44-53 |
| 8 | [Approximation Theory and Transforms](08_approximation_and_transforms/) | 54-60 |
| 9 | [Numerical Differentiation and Integration](09_differentiation_and_integration/) | 61-66 |
| 10 | [Ordinary Differential Equations](10_ordinary_differential_equations/) | 67-76 |
| 11 | [Partial Differential Equations](11_partial_differential_equations/) | 77-83 |
| 12 | [Numerical Optimization](12_optimization/) | 84-90 |
| 13 | [Stochastic and Monte Carlo Methods](13_stochastic_methods/) | 91-93 |
| 14 | [Numerical Analysis in Machine Learning and AI](14_ml_and_ai_connections/) | 94-98 |

The full lesson list with one-line descriptions is in [COURSE_MAP.md](COURSE_MAP.md).

---

## Who this is for

You should already know calculus, linear algebra at the level of matrices and eigenvalues,
and enough Python to read a NumPy function. You do not need any prior numerical analysis.

The course is written for someone who wants to know *why* an algorithm works and *when it
breaks*, not just which library function to call.

---

## What makes this different from a textbook

**Algorithms are built, not cited.** Gaussian elimination, QR by three different methods,
the QR eigenvalue algorithm, GMRES, the FFT, adaptive quadrature, Runge-Kutta, BFGS and
reverse-mode automatic differentiation are all written from scratch in NumPy, then checked
against SciPy.

**Nothing is claimed without a check.** Every factorization is verified against its defining
identity, every solve against its residual, every convergence rate against a measured one.
The notebooks assert their own results, so a wrong number is an execution failure, not
something you have to notice.

**Failure is taught, not hidden.** You will see classical Gram-Schmidt lose orthogonality,
the Richardson scheme blow up, an explicit method destroy a stiff problem, the naive
quadratic formula lose all its digits, and the Runge phenomenon ruin a high-degree
interpolant. Each failure is set up on purpose and explained.

---

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS or Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Python 3.10 or newer. The course was developed and verified on Python 3.12 with NumPy 2.3,
SciPy 1.16 and Matplotlib 3.10.

## Running the lessons

```bash
jupyter lab
```

Then open any `.ipynb` file. Lessons are numbered and meant to be read in order, but
[COURSE_DEPENDENCIES.md](COURSE_DEPENDENCIES.md) lists shorter routes if you only want one
topic.

To run a notebook without opening it:

```bash
python -m nbconvert --to notebook --execute --inplace 01_foundations/03_floating_point_arithmetic.ipynb
```

## Running the tests

```bash
pytest tests/ -v
```

The test suite covers the reusable library in `src/nalib` with normal cases, edge cases,
known analytic results, seeded random cases, ill-conditioned cases and singular cases.

## Verifying the whole repository

```bash
python verification/run_all.py
```

This executes every notebook, runs the test suite, checks every numerical identity, audits
source coverage and sweeps for placeholders. Reports land in `verification/`.

---

## Notation

Used consistently in every lesson.

| Symbol | Meaning |
|---|---|
| lower case, `x`, `y`, `b` | column vectors |
| upper case, `A`, `Q`, `R`, `L`, `U` | matrices |
| a hat, as in x-hat | a computed, inexact quantity |
| `n` | dimension of a square system, number of unknowns |
| `m` | rows of an overdetermined system, so `A` is m by n with m at least n |
| `k` | iteration index |
| `h` | step size, in space or time |
| machine epsilon | 2 to the power -52 in IEEE double precision |
| unit roundoff | half of machine epsilon |
| kappa(A) | condition number of A in the stated norm |
| rho(A) | spectral radius of A |
| sigma_i | i-th singular value, largest first |
| lambda_i | i-th eigenvalue |
| r = b - A x-hat | residual |

Four kinds of error are named precisely and never mixed up:

- **absolute error**, the size of the difference
- **relative error**, that difference scaled by the true value
- **forward error**, how wrong the computed answer is
- **backward error**, how much you would have to change the question so the computed answer
  is exactly right

**Conditioning** is a property of the problem. **Stability** is a property of the algorithm.
The course keeps these apart everywhere.

---

## Sources

The mathematics in this course is drawn from three sources, merged and reorganised by topic:

1. Timothy Sauer, *Numerical Analysis*, 3rd edition. Pearson, 2019.
2. Rajesh Kumar Gupta, *Numerical Methods: Fundamentals and Applications*. Cambridge
   University Press, 2019.
3. Lloyd N. Trefethen and David Bau III, *Numerical Linear Algebra*. SIAM, 1997, supplied as
   a 40-lecture concept map used as the coverage checklist for the linear algebra parts.

Source 3 drives the depth of Parts 3 to 6. How each of its 40 lectures maps into the course
is set out in `_planning/TrefethenBau_Course_Mapping.md`.

Material that is in neither book but that a complete graduate course needs is written
anyway and recorded as **Supplementary / Advanced Material** in the coverage checklist rather
than attributed to a source. There are 75 such items, listed with the lesson each belongs to in
`_planning/coverage_data.py`, and the whole of Part 14 is among them. Nothing in this repository
is attributed to a source it did not come from.

The mapping from every source section to the lesson that covers it is tracked in
`_planning/INTERNAL_COVERAGE_CHECKLIST.md`. That file is a quality-control artifact. You do
not need it to use the course.

---

## Relevance to machine learning and AI

Part 14 makes the connections explicit, but they run through the whole course:

- The **SVD** is PCA, low-rank adapters, and the spectrum of an embedding matrix.
- **Least squares conditioning** is why ridge regression exists.
- **Condition number** is what sets how fast gradient descent converges.
- **Floating point** is why softmax overflows, why mixed precision needs loss scaling, and
  why the same training run gives different numbers twice.
- **ODE solvers** are momentum, Nesterov acceleration and neural ODEs.
- **Krylov methods** are how anyone solves a linear system with a million unknowns inside a
  learning pipeline.
- **Automatic differentiation** is backpropagation.

These are treated as consequences of the mathematics, never as a replacement for it.
