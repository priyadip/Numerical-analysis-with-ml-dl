# Learning Path

How to actually work through this course, depending on what you want from it.

For the full lesson list see [COURSE_MAP.md](COURSE_MAP.md). For what depends on what, see
[COURSE_DEPENDENCIES.md](COURSE_DEPENDENCIES.md).

---

## Before you start

You need calculus, linear algebra up to eigenvalues, and enough Python to read a NumPy
function. Nothing else. Everything numerical is built from lesson 01.

```bash
pip install -r requirements.txt
jupyter lab
```

Then open [`01_foundations/01_what_is_numerical_analysis.ipynb`](01_foundations/01_what_is_numerical_analysis.ipynb).

## How to work through a lesson

Each lesson is a pair of files with the same name. Use whichever suits you:

- The **`.ipynb`** is the lesson to run. Execute it top to bottom once, then go back and
  change numbers. Break things on purpose.
- The **`.md`** is the same lesson to read, with the real output already in it. Use it on a
  train, or to review, or to search.

They contain the same mathematics and reach the same conclusions, because both are generated
from one source.

A lesson takes roughly 60 to 120 minutes to read carefully, and longer if you do the
exercises. Do not try to do more than one in a sitting.

**Do the exercises.** They are graded in five levels, and the levels are not decoration:

| Level | What it asks | Skip it if |
|---|---|---|
| 1, conceptual | Explain an idea in your own words | never skip these, they take two minutes |
| 2, mathematical | Derive or prove something | you are here for the code only |
| 3, computational | Implement an algorithm | you are here for the theory only |
| 4, experimental | Investigate a numerical phenomenon | you are short of time |
| 5, advanced | Open-ended, sometimes research adjacent | you are on a first pass |

Solutions are in [`solutions/`](solutions/), one file per part. Levels 1 and 2 are answered in
full. Levels 3 and 4 give the method and the answer you should get, so you can check yourself
rather than copy. Try the exercise before opening the solution: the value is in the attempt.

---

## Route 1: the whole course, in order

Lessons 01 through 98, straight through. This is the intended route and the only one that
guarantees every prerequisite is met when you need it.

| Part | Lessons | Roughly |
|---|---|---|
| 1, Foundations | 01 to 08 | 2 weeks |
| 2, Root finding | 09 to 14 | 1.5 weeks |
| 3, Direct linear systems | 15 to 22 | 2 weeks |
| 4, Iterative and Krylov | 23 to 28 | 2 weeks |
| 5, QR and least squares | 29 to 34 | 2 weeks |
| 6, Eigenvalues and SVD | 35 to 43 | 3 weeks |
| 7, Interpolation | 44 to 53 | 3 weeks |
| 8, Approximation and transforms | 54 to 60 | 2 weeks |
| 9, Differentiation and integration | 61 to 66 | 2 weeks |
| 10, ODEs | 67 to 76 | 3 weeks |
| 11, PDEs | 77 to 83 | 2.5 weeks |
| 12, Optimization | 84 to 90 | 2.5 weeks |
| 13, Stochastic methods | 91 to 93 | 1 week |
| 14, Machine learning connections | 94 to 98 | 1.5 weeks |

That is about 33 weeks at three lessons a week, which is two full semesters.

## Route 2: a one semester course

If you have about 14 weeks, this is the core. It keeps every idea that later material depends
on and drops the specialised branches.

```text
Weeks 1-2    01 02 03 04 05 06 07 08      foundations, all of it
Weeks 3-4    09 10 11 12                  root finding
Weeks 5-6    15 16 17 18 19 20 21         direct linear systems, with projectors
Weeks 7-8    23 24 25                     iterative methods and CG
Weeks 9-10   29 30 31 32                  least squares and QR
Weeks 11-12  35 36 37 41 43               eigenvalues and the SVD
Week 13      44 45 46 47                  interpolation
Week 14      61 62 63 65                  differentiation and quadrature
```

39 lessons. Add 67, 69 and 72 if the course needs to reach differential equations.

## Route 3: the fast foundation

Six lessons, about a week, and you will understand why numerical answers go wrong. This is
the minimum for anyone who computes with floating point at all, which is everyone.

```text
01  what the subject is, and problem versus algorithm
03  IEEE 754, machine epsilon, the standard model
04  the error taxonomy and how error propagates
05  cancellation, and how to rewrite a formula to avoid it
06  conditioning versus stability, and the diagnostic
08  what an algorithm actually costs
```

Skip 02 if you already know binary, and 07 if your calculus is fresh.

---

## Goal-directed routes

Each of these assumes you have done Route 3 first, or already know that material.

### I want to solve linear systems properly

```text
15 norms  ->  17 elimination and LU  ->  18 pivoting  ->  19 conditioning
           ->  20 Cholesky  ->  21 sparse and banded  ->  22 condition estimation
```

Then 23 to 25 if your matrices are large and sparse, and 28 if they come from a PDE.

### I want to understand the SVD and PCA

```text
15 norms  ->  16 orthogonality and projectors  ->  29 least squares
           ->  30 Gram-Schmidt  ->  31 Householder  ->  35 eigenvalue theory
           ->  41 SVD theory  ->  43 low rank  ->  94 machine learning connections
```

Lesson 42 as well if you want to know how the SVD is actually computed.

### I want the eigenvalue algorithms end to end

```text
15 norms  ->  16 orthogonality and projectors  ->  30 QR  ->  31 Householder
           ->  35 theory and localization  ->  36 power methods
           ->  37 the QR algorithm  ->  38 symmetric problems  ->  39 Krylov eigensolvers
```

Add 40 for the generalized problem $Ax = \lambda Bx$.

### I want to solve differential equations

```text
06 stability  ->  07 convergence  ->  62 quadrature  ->  67 IVP theory and Euler
              ->  69 Runge-Kutta  ->  70 adaptive steps  ->  72 stiffness
              ->  73 systems  ->  75 boundary value problems
```

Then Part 11 for PDEs, which also needs 17 and 21 for the linear solves.

### I want the numerical side of machine learning

```text
03 floating point  ->  05 cancellation  ->  06 conditioning  ->  15 norms
   ->  16 orthogonality and projectors  ->  24 conjugate gradient
   ->  29 least squares  ->  41 SVD theory
   ->  84 optimization fundamentals  ->  86 gradient methods
   ->  89 stochastic optimization  ->  94 linear algebra in ML
   ->  95 optimization for ML  ->  96 floating point in deep learning
   ->  97 automatic differentiation
```

Lesson 96 is the one that explains why your training run gives different numbers twice.

### I am debugging a numerical result right now

Read [`01_foundations/06_conditioning_and_stability.md`](01_foundations/06_conditioning_and_stability.md),
section 9. It is a four-step procedure that tells you whether the problem or your code is at
fault. Then follow whichever branch it sends you down.

---

## How to tell whether you have understood a lesson

Three checks, in increasing order of confidence.

1. **Can you state the main result without looking?** If a lesson has a boxed inequality or a
   named theorem, that is the thing.
2. **Can you predict the experiment before running it?** Read the setup, guess the shape of
   the plot, then run it. Getting this wrong is the most useful thing that can happen to you.
3. **Can you break it?** Change a parameter until the method fails, and explain the failure
   using the lesson's own vocabulary. Every lesson has a failure mode, and several teach it
   deliberately.

## A note on how to read the experiments

Every numerical experiment in this course asserts its own conclusions, so if a notebook runs,
its claims held on your machine too. When a lesson says a measured value matched a prediction,
there is an `assert` a few lines below making that a fact rather than a claim.

Where a measurement does **not** match the naive prediction, the lesson says so and explains
why. Lesson 08 is the clearest case: matrix multiplication is $O(n^3)$ in arithmetic but times
out at an exponent near 2.3, and the whole gap is accounted for by the machine getting more
efficient on bigger problems. That is a more useful thing to learn than a number that was
forced to agree.
