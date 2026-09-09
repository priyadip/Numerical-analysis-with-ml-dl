# Curriculum Change Log

Every structural change to the course after the Phase 1 design is recorded here, with the
reason, the full effect, and what had to be regenerated.

---

## Change 001: insert lesson 16, "Orthogonality and Projectors"

**Date:** 2026-08-20
**Trigger:** Integration audit of `TrefethenBau.md`, recorded in
[`TrefethenBau_Course_Mapping.md`](TrefethenBau_Course_Mapping.md)
**Type:** one lesson inserted, all higher lesson numbers shift by one
**Course size:** 95 lessons becomes **96 lessons**. Parts stay at 14.

### Why

Two Trefethen and Bau lectures had no home anywhere in the 95-lesson plan:

- **L2, Orthogonal Vectors and Matrices**
- **L6, Projectors**

They are one idea, and it is the idea the book is built on. Orthogonality gives projectors,
projectors give QR, QR gives least squares, and the identical projection idea reappears as
Rayleigh-Ritz in the Krylov chapters. Without it, three later parts of this course teach what
look like three unrelated tricks.

The audit tested this against the four conditions the brief sets for adding a lesson:

| Test | Answer |
|---|---|
| Is the concept important? | Yes. Six later lessons depend on it directly. |
| Can it fit inside an existing lesson? | No good option. See below. |
| Would combining overload the host? | Yes. Lesson 15 would carry norms, orthogonality and projectors at once. |
| Does it deserve standalone treatment? | Yes. Own theorems, own geometry, own experiments. |

**Options considered and rejected:**

- *Fold into lesson 15 (norms).* Rejected. That lesson is already a prerequisite for everything
  in Parts 3 to 6, and tripling its scope makes it the bottleneck of the whole course.
- *Fold into lesson 30 (Gram-Schmidt and QR).* Rejected on ordering. Part 4 (Krylov) comes
  before Part 5 (QR) in this course, a deliberate choice recorded in `COURSE_ARCHITECTURE.md`
  section 3. Arnoldi and Rayleigh-Ritz are projection methods, so projectors must exist before
  Part 4 starts.
- *Reorder the course to put QR before Krylov.* Rejected. That would undo a considered
  pedagogical decision and move six lessons, to solve a problem that one insertion solves.
- *Leave it out.* Rejected. It is the conceptual hinge of the reference document.

### Position

Part 3, Direct Methods for Linear Systems, immediately after lesson 15.

The two ideas are neighbours: lesson 15 says norms measure length, lesson 16 says orthogonal
matrices preserve it. That single fact, $\|Qx\|_2 = \|x\|_2$ and hence $\kappa_2(Q) = 1$, is why
every stable algorithm later in the course is assembled from reflections and rotations, and the
course previously never stated it.

### Cost of making this change now

**Near zero.** All eight written lessons are numbered 01 to 08, below the insertion point, so:

- no written `.ipynb` was touched
- no written `.md` was touched
- no figure was regenerated
- no test changed
- no coverage flag was lost

Only planning metadata and generated documents changed. Making the same change after Part 3 was
written would have required renumbering and re-executing every notebook above lesson 15.

### Renumbering

| Old | New | Lesson |
|---|---|---|
| 01 to 15 | unchanged | Part 1 and lesson 15 |
| **new** | **16** | **orthogonality_and_projectors** |
| 16 to 95 | 17 to 96 | everything from Gaussian elimination onward, shifted by one |

Part ranges after the change:

| Part | Name | Before | After | Count |
|---|---|---|---|---|
| 1 | Foundations of Numerical Computing | 01-08 | 01-08 | 8 |
| 2 | Nonlinear Equations and Root Finding | 09-14 | 09-14 | 6 |
| 3 | Direct Methods for Linear Systems | 15-21 | **15-22** | **8** |
| 4 | Iterative and Krylov Subspace Methods | 22-27 | 23-28 | 6 |
| 5 | Orthogonality, QR and Least Squares | 28-33 | 29-34 | 6 |
| 6 | Eigenvalue Problems and the SVD | 34-42 | 35-43 | 9 |
| 7 | Interpolation | 43-52 | 44-53 | 10 |
| 8 | Approximation Theory and Transforms | 53-59 | 54-60 | 7 |
| 9 | Numerical Differentiation and Integration | 60-65 | 61-66 | 6 |
| 10 | Ordinary Differential Equations | 66-75 | 67-76 | 10 |
| 11 | Partial Differential Equations | 76-82 | 77-83 | 7 |
| 12 | Numerical Optimization | 83-87 | 84-88 | 5 |
| 13 | Stochastic and Monte Carlo Methods | 88-90 | 89-91 | 3 |
| 14 | Numerical Analysis in Machine Learning and AI | 91-95 | 92-96 | 5 |

Numerical linear algebra, Parts 3 to 6, goes from 28 lessons to **29 of 96**.

Note on naming: Part 5 keeps the title "Orthogonality, QR and Least Squares" even though
orthogonality itself now sits in Part 3. The Part 5 title describes what the learner does
there, which is use orthogonality to solve least squares problems. Changing it would rename a
folder for no benefit.

### Prerequisites and dependencies

**Lesson 16 requires:** 06 (conditioning and stability), 15 (norms)

**Lesson 16 unlocks:** 24, 26, 27, 29, 30, 31, 32, 39, 41

New edges added to the dependency graph:

```
15 --> 16
06 --> 16
16 --> 24    A-orthogonality in the CG inner product
16 --> 26    orthogonalizing the Krylov basis
16 --> 29    least squares IS a projection
16 --> 30    Gram-Schmidt applies projectors
16 --> 31    reflectors are orthogonal transformations
16 --> 39    Rayleigh-Ritz is a projection method
16 --> 41    the SVD is an orthogonal change of basis
```

### Trefethen and Bau concepts this lesson covers

L2 (Orthogonal Vectors and Matrices) and L6 (Projectors), in full. Those are the only two
lectures in the reference document with status REQUIRES NEW LESSON.

### Files changed

| File | Change | How |
|---|---|---|
| `_planning/lesson_map.py` | one entry inserted, 80 renumbered | edited |
| `_planning/coverage_data.py` | all lesson ids above 15 shifted, new rows for L2 and L6 concepts | edited |
| `_planning/coverage_status.json` | rebuilt, Part 1 flags preserved | regenerated |
| `_planning/INTERNAL_COVERAGE_CHECKLIST.md` | rebuilt | regenerated |
| `COURSE_MAP.md` | rebuilt | regenerated |
| `COURSE_DEPENDENCIES.md` | lesson numbers and new edges | edited |
| `LEARNING_PATH.md` | all route listings above lesson 15 | edited |
| `COURSE_ARCHITECTURE.md` | part table, phase table, lesson counts | edited |
| `README.md` | part ranges, phase table | edited |
| `verification/*` | rebuilt | regenerated |

### Verification after the change

`python verification/run_all.py` must still report all checks passing, with:

- 8 of 8 notebooks passing, unchanged
- 219 of 219 tests passing, unchanged
- 0 broken links
- concept count rising to reflect the new lesson's mapped concepts
- Part 1 coverage flags **unchanged**, since no Part 1 lesson was touched

**Result, run 2026-08-20 after the change:**

```
[1/7] notebooks    : 8 passed, 0 failed          unchanged
[2/7] test suite   : 219 passed, 0 failed        unchanged
[3/7] numerical    : 13/13 identities, 67 asserts unchanged
[4/7] coverage     : 49 of 507 concepts covered   was 47 of 449
[5/7] pairing      : 0 problems
[6/7] links        : 0 broken
[7/7] placeholders : 0 found
ALL CHECKS PASSED
```

Part 1 lost nothing. The concept total rose from 449 to 507 because the audit added 56
Trefethen and Bau rows and 2 supplementary rows, and reclassified 19 rows from SUP to S3. The
covered count rose from 47 to 49 because two of the reclassified rows belong to already-written
lessons 03 and 06.

---

## Change 002: promote Gauss quadrature and Lanczos to a core connection

**Date:** 2026-08-20
**Trigger:** Same audit. Trefethen and Bau L37 is a full lecture; the plan had Golub-Welsch
marked as supplementary.
**Type:** scope change to one lesson, no renumbering

Lesson 64 (post-change **65**), Gaussian Quadrature, moves the Golub-Welsch algorithm and the
Lanczos connection from supplementary to **core**, and gains a dependency on lesson 26
(Arnoldi and Lanczos).

Reason: the Lanczos tridiagonal matrix *is* the Jacobi matrix of the orthogonal polynomials.
Its eigenvalues are the Gauss nodes and the squared first components of its eigenvectors are
the weights. Computing a quadrature rule and computing eigenvalues of a sparse matrix turn out
to be the same computation. Leaving that as an optional aside wastes one of the best
connections in the whole subject.

New dependency edge: `26 --> 65`.

---

## Change 003: two modern sections added

**Date:** 2026-08-20
**Trigger:** Same audit, section 22 of the integration brief
**Type:** two new sections inside existing lessons, no renumbering

| Addition | Lesson | Classification | Reason |
|---|---|---|---|
| Matrix-free operators | 26 | CORE | Krylov methods need only the ability to compute $Av$, never $A$ itself. That is the whole reason they scale, and it deserves to be said rather than implied. |
| Sketching for least squares | 43 | MODERN SUPPLEMENT | Natural companion to the randomized SVD already planned there. |

Both are labelled in the lesson as not being from the supplied sources.

---

---

## Change 004: incidental corrections found during the renumbering

**Date:** 2026-08-20
**Type:** corrections, no structural effect

Renumbering `LEARNING_PATH.md` surfaced two numbers that were wrong before the change:

| Statement | Was | Now | Why |
|---|---|---|---|
| Route 2 lesson count | 48 lessons | 39 lessons | The original figure was never correct. The count is now derived from the listing itself. |
| Route 1 total duration | 29 weeks | 32 weeks | Recomputed for 96 lessons at three a week. |

Also added lesson 16 to the Route 2 core listing and to the SVD, eigenvalue, sparse solver and
machine learning goal routes, since projectors are a prerequisite for all four.

---

## Change 005: two lessons added to Part 12, "Stochastic Optimization" and "Proximal and Composite Optimization"

**Date:** 2026-09-06
**Trigger:** the mathematical treatment of the stochastic optimizers had no home in Part 12, and
the only place they appeared was lesson 93 in the machine learning part, where they would have
had to be taught and applied in the same lesson.
**Type:** two lessons appended to Part 12, all higher lesson numbers shift by two
**Course size:** 96 lessons becomes **98 lessons**. Parts stay at 14.

### Why

Part 12 as planned covered deterministic smooth optimization only: fundamentals, derivative free
search, gradient and Newton methods, quasi-Newton and trust regions, and constraints. Two families
that a graduate treatment cannot leave out were missing entirely.

**The stochastic family.** SGD, minibatch gradient noise, momentum, Nesterov, AdaGrad, RMSProp,
Adam and AdamW are optimization methods with their own convergence theory, their own step size
rules and their own failure modes. They belong next to the deterministic methods they modify, not
in an applications part, and the reason is that the interesting content is mathematical: what the
noise does to the convergence rate, why a decreasing step size is needed for convergence and a
constant one is used anyway, and why the adaptive methods are diagonal preconditioners in disguise.

**The composite family.** A great many real objectives are `smooth + nonsmooth`, and nothing in
Parts 1 to 12 could handle one. The proximal operator, proximal gradient descent, ISTA and FISTA,
and soft thresholding for L1 and LASSO are the standard answer, they connect directly to lesson
88's projected gradient (projection is the proximal operator of an indicator function), and FISTA
is Nesterov acceleration from lesson 89 applied to it.

### What lesson 95 keeps

The old lesson 93, now 95, listed "Gradient descent as an ODE discretization, momentum and
Nesterov, SGD, Adam, why second-order methods stay rare". Teaching the optimizers and their
machine learning consequences in one lesson would have meant deriving Adam and discussing
generalization in the same forty pages.

The split is: **Part 12 lesson 89 teaches the methods, Part 14 lesson 95 measures what they do to
a learning problem.** Lesson 95 keeps gradient descent as an ODE discretization, natural gradient
and Gauss-Newton, and gains sharp and flat minima, batch size and generalization, which is the
question that is genuinely about learning rather than about optimization.

### The renumbering

| Old | New | Lesson |
|---|---|---|
| new | 89 | Stochastic Optimization |
| new | 90 | Proximal and Composite Optimization |
| 89 | 91 | Random Number Generation |
| 90 | 92 | Monte Carlo Methods |
| 91 | 93 | Brownian Motion and SDEs |
| 92 | 94 | Numerical Linear Algebra in Machine Learning |
| 93 | 95 | Optimization for Machine Learning |
| 94 | 96 | Floating Point in Deep Learning |
| 95 | 97 | Automatic Differentiation |
| 96 | 98 | Scientific Machine Learning and Inverse Problems |

Part spans become: Part 12 lessons 84 to 90, Part 13 lessons 91 to 93, Part 14 lessons 94 to 98.

### Cost

**Zero written lessons were renumbered.** Lessons 84 and 85 are written and keep their numbers;
everything that moved is unwritten, which is exactly the condition standing rule 1 asks for.

### Concepts

Seven tracked concepts added and one moved, so the checklist goes from **507 to 514**.

| Concept | Lesson |
|---|---|
| Stochastic and minibatch gradient descent, and gradient noise | 89, new |
| Momentum, Nesterov acceleration and the heavy ball method | 89, moved from 93 and split |
| Adaptive step sizes: AdaGrad, RMSProp, Adam and AdamW | 89, new |
| Learning rate schedules and the convergence rate of SGD | 89, new |
| The proximal operator and proximal gradient descent | 90, new |
| ISTA and FISTA acceleration for composite problems | 90, new |
| L1 regularization, LASSO and the soft thresholding operator | 90, new |
| Sharp and flat minima, batch size and generalization | 95, new |

All eight are supplementary. Neither supplied source covers stochastic or proximal optimization,
which is recorded rather than hidden: Sauer stops at Nelder-Mead and Gupta has no optimization
chapter at all.

### Incidental corrections

The renumbering surfaced four cross references that were **already wrong** before this change,
left over from change 001:

| File | Said | Should have said | Now says |
|---|---|---|---|
| lesson 03 | non-deterministic reductions, lesson 93 | 94 | 96 |
| lesson 04 | automatic differentiation, lesson 94 | 95 | 97 |
| lesson 05 | non-determinism, lesson 93 | 94 | 96 |
| lesson 12 | automatic differentiation, lesson 95 | 95 | 97 |

Lessons 03, 04, 05 and 12 were rebuilt and re-executed.

### Files touched

Hand edited: `lesson_map.py`, `coverage_data.py`, `COURSE_ARCHITECTURE.md`, and the four lesson
sources above.

Regenerated: `COURSE_MAP.md`, `INTERNAL_COVERAGE_CHECKLIST.md`, `coverage_status.json`, the four
rebuilt notebooks and their markdown, and everything in `verification/`.

---

## Standing rules for future changes

1. Renumbering is cheapest before the affected lessons are written. Do structural changes early
   or not at all.
2. Any change that would renumber a **written** lesson must be justified far more strongly than
   this one, because it invalidates executed notebooks, figures and coverage flags.
3. Every change gets an entry here with: trigger, options rejected, cost, full renumbering
   table, dependency edges, files touched, and verification result.
4. Generated files (`COURSE_MAP.md`, the coverage checklist, everything in `verification/`) are
   never edited by hand. They are regenerated.
