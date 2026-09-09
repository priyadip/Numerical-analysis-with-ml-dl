# Trefethen and Bau, Integration Mapping into the Existing Course

Reference document: [`D:\Numerical analysis\TrefethenBau.md`](../../TrefethenBau.md), 40 lectures
in 6 parts. That file is the authoritative coverage checklist and is **not** reproduced here.
This document says only **where each of its concepts goes** in the existing course, **how deep
the treatment must be**, and **what is currently missing**.

Guiding rule applied throughout: **preserve, integrate, strengthen, add** in that order. The
existing 14-part structure is treated as the default and is kept.

---

## 1. Executive summary

**Verdict: the existing course absorbs Trefethen and Bau almost entirely, with one genuine
structural gap.**

| Outcome | T&B lectures | Share |
|---|---|---|
| Already fully covered by an existing lesson's plan | 11 | 27% |
| Covered but needs strengthening (PARTIAL or SHALLOW) | 26 | 65% |
| Genuinely missing, no home in the current plan | 3 | 8% |

The three homeless lectures are **L2 Orthogonal Vectors and Matrices**, **L6 Projectors**, and
the projection-method framing that L34 and L38 both lean on. They are one coherent idea, and
they are the hinge of the book: orthogonality gives projectors, projectors give QR, QR gives
least squares, and the same projection idea reappears as Rayleigh-Ritz in the Krylov chapters.

**One new lesson is therefore proposed**, and only one:

> **New lesson 16, "Orthogonality and Projectors"**, inserted in Part 3 immediately after
> lesson 15 (norms) and before the Gaussian elimination material. All later lessons shift by
> one, giving 96 lessons instead of 95.

Justification against the four tests in the brief, and the renumbering table, are in section 8
and in [`CURRICULUM_CHANGE_LOG.md`](CURRICULUM_CHANGE_LOG.md).

Everything else is handled by strengthening lessons that already exist. No lesson is deleted,
no part is reorganised, and the eight written lessons of Part 1 need **no renumbering** because
they all sit below the insertion point. This is the cheapest possible moment to make this
change.

Three further consequences of the audit, beyond simple placement:

1. **The column viewpoint is missing from the whole course.** T&B opens by insisting that $Ax$
   is a linear combination of the columns of $A$, and that $AB$ is a matrix whose columns are
   $A$ times the columns of $B$. This reframing is used silently throughout the book and it is
   currently nowhere in the plan. It goes into lesson 15.
2. **Stability of triangular solves is missing.** T&B devotes a whole lecture to the fact that
   back substitution is backward stable and that triangular systems are usually far more
   accurately solvable than their condition number suggests. This is a genuinely surprising
   result and it is currently absent. It goes into lesson 17.
3. **The Lanczos to Gauss quadrature connection is planned as an afterthought.** Lesson 65
   currently lists Golub-Welsch as supplementary. T&B makes it a full lecture connecting
   Krylov methods, orthogonal polynomials and quadrature. It is promoted to core.

---

## 2. Concepts already fully covered

These T&B lectures land cleanly on an existing lesson whose planned scope already matches, and
need integration only, not strengthening.

| T&B lecture | Lands in | Note |
|---|---|---|
| L13 Floating-Point Arithmetic | **03** (written) | Verified complete: representation, model, machine precision, unit roundoff, rounding, underflow, overflow. Cancellation is deliberately split to lesson 05 and accumulation to lesson 04. |
| L9 Numerical Computing Environment | **01, 08** (written) plus repository infrastructure | Vectorised computation, verification and reproducibility are how the whole repository is built, not just a topic in it. |
| L20 Gaussian Elimination | **17** | Elimination, LU, operation counts, complexity all already planned. |
| L21 Pivoting | **18** | Partial and complete pivoting, permutation matrices, PA=LU, growth factor already planned. |
| L23 Cholesky Factorization | **20** | Positive definiteness, cost, comparison with LU already planned. |
| L24 Eigenvalue Problems | **35** | Similarity, diagonalizability, defective matrices, multiplicities, Schur form, eigenvalue conditioning already planned. |
| L26 Reduction to Hessenberg or Tridiagonal | **37, 38** | Hessenberg in 37, tridiagonalization by Givens and Householder in 38. |
| L28 QR Algorithm Without Shifts | **37** | Simultaneous iteration and the unshifted algorithm already planned, and confirmed present in Sauer p566. |
| L29 QR Algorithm With Shifts | **37** | Shifts, deflation, real Schur form already planned. |
| L30 Other Eigenvalue Algorithms | **38** | Jacobi rotations, bisection by Sturm sequence, divide and conquer already planned. |
| L31 Computing the SVD | **42** | Bidiagonalization, Golub-Kahan, implicit QR, why forming $A^*A$ is wrong, all already planned. |

**11 lectures. No action needed beyond writing the lesson to its existing plan.**

---

## 3. Concepts partially covered

The concept is planned, but a specific and important piece of it is absent. Each row names the
missing piece.

| T&B lecture | Lands in | What is planned | What is missing |
|---|---|---|---|
| L1 Matrix-Vector Multiplication | **15**, cost in **08** | Norms, matrix basics; cost is fully done in 08 | **The column viewpoint.** $Ax$ as a linear combination of columns, $AB$ column by column, and the geometric reading of a matrix as a linear map. This is T&B's organising idea. |
| L3 Norms | **15** | Vector and matrix norms, induced norms, equivalence, submultiplicativity, spectral radius | Norm-based **error bounds** as a working tool, and the explicit link from the induced 2-norm to $\sigma_1$ (forward reference to 41) |
| L4 Singular Value Decomposition | **41** | Geometry, existence, four subspaces, pseudoinverse | **Conditioning through singular values**, meaning $\kappa_2(A) = \sigma_1/\sigma_n$ derived rather than asserted, and minimum-norm solutions |
| L5 More on the SVD | **43** | Numerical rank, truncated SVD, Eckart-Young-Mirsky | **Singular value perturbation**, meaning the Weyl inequalities and why singular values are perfectly conditioned |
| L7 QR Factorization | **30** | Reduced and full QR, Gram-Schmidt, cost | **QR expressed through projectors** (depends on the new lesson 16), and uniqueness conditions |
| L11 Least Squares Problems | **29, 32** | Normal equations, model fitting, pseudoinverse, rank deficiency | **The projector formulation** $P = A(A^*A)^{-1}A^*$ and the three-way equivalence of normal equations, QR and SVD as the same projection |
| L12 Conditioning and Condition Numbers | **06** (written), **19** | Relative condition number, forward and backward error, the governing inequality | **Absolute** conditioning as distinct from relative. Currently only relative is defined. Small but it is a definition T&B uses. |
| L22 Stability of Gaussian Elimination | **18** | Growth factor, partial and complete pivoting | **The worst case versus practice paradox**: growth can be $2^{n-1}$, yet partial pivoting works in practice and nobody fully knows why |
| L25 Overview of Eigenvalue Algorithms | **35** | Eigenvalue theory | **Why eigenvalue computation must be iterative.** The Abel and Galois argument that no finite rational algorithm can exist for $n \ge 5$. This is the reason the entire eigenvalue part looks different from the linear systems part, and it is currently unstated. |
| L27 Rayleigh Quotient and Inverse Iteration | **36** | Power, inverse power, shifts, Rayleigh quotient iteration | **Why the Rayleigh quotient is accurate to second order** at an eigenvector, which is what makes RQI cubic rather than quadratic |
| L32 Overview of Iterative Methods | **23** | Splitting methods, spectral radius convergence | The **direct versus iterative decision** as a reasoned choice driven by size, sparsity and required accuracy |
| L33 Arnoldi Iteration | **26** | Krylov subspaces, Arnoldi, Lanczos, loss of orthogonality | The **Arnoldi relation** $AQ_m = Q_{m+1}\tilde{H}_m$ written explicitly and verified numerically. Everything in GMRES and Krylov eigensolvers rests on this one identity. |
| L35 GMRES | **27** | Residual minimisation, Givens rotations, restarting | **Stagnation**, and the fact that GMRES convergence is not governed by eigenvalues alone for nonnormal matrices |
| L36 Lanczos Iteration | **26** | Three-term recurrence, tridiagonal, loss of orthogonality | **Ghost eigenvalues** and the reorthogonalization strategies (full, selective, none) with the cost trade-off |
| L38 Conjugate Gradients | **24** | A-inner product, conjugacy, convergence bound in $\sqrt\kappa$ | **CG is Lanczos.** The two are the same recurrence viewed differently. Also the effect of finite precision on the finite-termination property. |
| L40 Preconditioning | **25** | Jacobi, SSOR, incomplete Cholesky, preconditioned CG | **Left, right and split** preconditioning as distinct operations, and **eigenvalue clustering** as the real goal rather than condition number reduction |

**16 lectures. Action: write the lesson to plan, then add the named missing piece.**

---

## 4. Concepts too shallow

The topic appears in the plan but at a depth below what a graduate treatment requires. These
need a section, an experiment, or a derivation, not a sentence.

| T&B lecture | Lands in | Why the current plan is too shallow | Required depth |
|---|---|---|---|
| L8 Gram-Schmidt | **30** | The plan says "the loss-of-orthogonality experiment", which is the demonstration but not the explanation | Derive **why** classical Gram-Schmidt loses orthogonality and modified does not, in terms of which quantity each subtracts a projection from. Measure $\|Q^*Q - I\|$ against $\kappa(A)$ across several decades and show the CGS line has slope 2 and the MGS line slope 1. |
| L10 Householder Triangularization | **31** | The plan says "the sign trick" without saying why | Full construction of the reflector, the reflection geometry, **why the sign choice avoids cancellation** (it is lesson 05 applied to $v = x \pm \|x\|e_1$), the flop count, and why $Q$ is stored as reflectors rather than formed |
| L14 Stability | **06** (written) | Framework is complete and correct; a worked backward error analysis is only in the solutions | Promote a full worked backward error analysis into the lesson body. Currently the Horner proof lives in `solutions/part01_foundations.md` exercise 5.2. |
| L15 More on Stability | **06** (written), **31** | Only the framework exists | A **detailed** backward error analysis of a real algorithm, carried through the floating point model term by term. Householder QR in lesson 31 is the natural place. |
| L16 Stability of Householder Triangularization | **31** | Plan says "stability comparison" only | State and verify the backward stability result. The measured product $\tilde{Q}\tilde{R}$ must reproduce $A$ to machine precision **even when $\tilde{Q}$ has lost orthogonality**. That gap between the two is the entire point and it is a striking experiment. |
| L17 Stability of Back Substitution | **17** | **Absent.** Triangular solves are mentioned as an operation count only | Back substitution is backward stable, with the surprisingly strong componentwise result. Triangular systems are usually solved far more accurately than $\kappa$ predicts. Needs its own section plus an experiment on a badly conditioned triangular matrix. |
| L18 Conditioning of Least Squares | **32** | Plan mentions the squared condition number only | The **four** condition numbers, sensitivity of $x$ and of $y = Ax$ to perturbations in $A$ and in $b$ separately, and the role of the angle $\theta$ between $b$ and $\mathrm{range}(A)$. The squared condition number is a consequence, not the statement. |
| L19 Stability of Least Squares Algorithms | **32** | Plan says "normal equations versus QR versus SVD on the same hard problem" | Say **why** normal equations lose: forming $A^*A$ squares the condition number, and this is a property of the algorithm, not of the problem. Householder QR is backward stable for least squares, Gram-Schmidt is not unless the right variant is used. Measure all four against a high precision reference. |
| L34 How Arnoldi Locates Eigenvalues | **39** | Plan says "Ritz value convergence" | The **Rayleigh-Ritz procedure** stated as a projection method, which is exactly the new lesson 16 material reappearing. Ritz values and vectors, their residuals as a stopping test, and which part of the spectrum converges first and why. |
| L37 From Lanczos to Gauss Quadrature | **65** | Golub-Welsch is currently marked supplementary | Promote to core. The Lanczos tridiagonal matrix **is** the Jacobi matrix of the orthogonal polynomials, its eigenvalues are the Gauss nodes and the squared first components of its eigenvectors are the weights. Implement Golub-Welsch, verify the nodes against `numpy.polynomial.legendre.leggauss`. |
| L39 Biorthogonalization | **27** | BiCGSTAB is a passing mention | Two-sided Lanczos, biorthogonality of the two Krylov bases, the short recurrence that makes BiCG cheap, and **breakdown**, which is the price of that short recurrence. State the short-recurrence versus optimality trade-off: GMRES is optimal but grows, BiCG is cheap but can break down. |

**11 lectures. Action: substantial new sections, derivations and experiments.**

---

## 5. Concepts missing

Three items with no home in the current plan.

| T&B lecture | Status | Resolution |
|---|---|---|
| **L2 Orthogonal Vectors and Matrices** | MISSING | New lesson 16. Inner products, orthonormal sets, orthogonal and unitary matrices, and the single most important numerical fact about them: they preserve the 2-norm, so they cannot amplify error. That fact is the reason every stable algorithm in the book is built from them, and the course currently never says it. |
| **L6 Projectors** | MISSING | New lesson 16. $P^2 = P$, orthogonal projectors $P^* = P$, complementary projectors, $P = QQ^*$ for orthonormal $Q$, rank-one projectors, projection error and residual orthogonality. |
| **Projection methods as a unifying frame** | MISSING | New lesson 16 introduces it; lessons 29, 32 and 39 then reuse it. Least squares is a projection, QR builds the projector, and Rayleigh-Ritz is the same projection applied to an eigenvalue problem. Without lesson 16 these three look like unrelated tricks. |

Nothing else in the reference document is unaccounted for.

---

## 6. Chapter-by-chapter integration plan

Lesson numbers below are **post-renumbering** (see section 8 and the change log). Only lessons
that change are listed. Lessons 01 to 08 are already written; changes to them are marked
**RETROFIT** and are small.

### Lesson 06, Conditioning and Stability (WRITTEN, retrofit)

- **T&B concepts to integrate:** L12 (absolute conditioning), L14 (complete), L15 (worked
  analysis)
- **Currently covered:** relative condition number, forward and backward error, the governing
  inequality, backward stability, the four cases, the diagnostic procedure, the $\kappa u$ rule
  of thumb. All verified.
- **Missing:** absolute condition number as a separate definition
- **Too shallow:** a worked backward error analysis appears only in the solutions file
- **New sections required:** a short subsection distinguishing absolute from relative
  conditioning; promote the Horner backward error analysis from the solutions into the lesson
  body as a worked example
- **New experiments:** none, the lesson already has three
- **Prerequisites:** unchanged
- **Later dependencies:** 15, 16, 19, 29, 31, 32, 35, 41

### Lesson 15, Vectors, Matrices and Norms (PLANNED, strengthen)

- **T&B concepts to integrate:** L1, L3
- **Currently planned:** vector and matrix norms, induced norms, equivalence,
  submultiplicativity, spectral radius
- **Missing:** the column viewpoint of $Ax$ and $AB$; norm-based error bounds as a tool
- **New sections required:**
  1. *A matrix is a linear map, and $Ax$ is a combination of columns.* Includes $AB$ read
     column by column, and the row and outer-product views as alternatives.
  2. *Norm-based error bounds.* How $\|A\|$ turns a bound on the input into a bound on the
     output, which is the machinery every later error bound uses.
- **New experiments:** show the three interpretations of $AB$ (inner product, column, outer
  product) producing identical results and different memory access patterns, tying back to
  lesson 08
- **Prerequisites:** 06
- **Later dependencies:** everything in Parts 3 to 6

### Lesson 16, Orthogonality and Projectors (NEW)

Full specification in section 8.

### Lesson 17, Gaussian Elimination and LU (PLANNED, strengthen)

- **T&B concepts to integrate:** L20 (fully), **L17 (currently absent)**
- **Currently planned:** naive elimination, substitution, operation counts, LU, Doolittle,
  Crout, Gauss-Jordan, why Cramer fails
- **Missing:** the entire stability analysis of triangular solves
- **New sections required:** *Stability of back substitution.* Back substitution is backward
  stable: the computed $\hat{x}$ solves exactly a system $(R + \delta R)\hat{x} = b$ with
  $|\delta R| \le n u |R|$ componentwise. Consequence: triangular systems are often solved to
  far better accuracy than $\kappa(R)$ suggests, because the componentwise bound is much
  stronger than the normwise one.
- **New experiments:** build a triangular matrix with $\kappa \approx 10^{12}$, solve it, and
  measure the forward error. It will be far better than $\kappa u$. Then measure the backward
  error and find it at machine precision. This is a clean instance of lesson 06's diagnostic.
- **Prerequisites:** 06, 15
- **Later dependencies:** 18, 19, 20, 30, 31

### Lesson 18, Pivoting and PA=LU (PLANNED, strengthen)

- **T&B concepts to integrate:** L21 (fully), L22
- **Missing:** the worst case versus practice paradox
- **New sections required:** *Why partial pivoting works, and why nobody fully knows.* The
  growth factor can reach $2^{n-1}$, Wilkinson's matrix attains it, and yet partial pivoting is
  the universal default because growth of that size essentially never occurs on real data.
  Complete pivoting has a proven polynomial bound and is not used, because the extra $O(n^3)$
  comparisons are not worth it.
- **New experiments:** construct the Wilkinson growth matrix and measure the actual growth
  factor reaching $2^{n-1}$; then measure growth on 10000 random matrices and show it stays
  near $n^{2/3}$. The gap between worst case and typical is the lesson.
- **Prerequisites:** 17
- **Later dependencies:** 19, 20

### Lesson 19, Conditioning of Linear Systems (PLANNED, strengthen)

- **T&B concepts to integrate:** L12 applied to $Ax = b$
- **Missing:** derivation of $\kappa_2(A) = \sigma_1/\sigma_n$ (forward reference to 41)
- **New sections required:** the perturbation theorem for $Ax = b$ derived, not quoted, for
  perturbations in $b$ and in $A$ separately
- **Prerequisites:** 06, 15, 17
- **Later dependencies:** 22, 24, 29, 32

### Lesson 23, Classical Iterative Methods (PLANNED, strengthen)

- **T&B concepts to integrate:** L32
- **Missing:** the direct versus iterative decision as a reasoned choice
- **New sections required:** *When to stop factorizing and start iterating.* Driven by three
  numbers: the size $n$, the sparsity, and the accuracy actually needed. Reuses the memory
  table from lesson 08.
- **Prerequisites:** 15, 17, 21
- **Later dependencies:** 24 to 28

### Lesson 24, Conjugate Gradient (PLANNED, strengthen)

- **T&B concepts to integrate:** L38
- **Missing:** CG and Lanczos are the same recurrence; finite precision destroys finite
  termination
- **New sections required:** *CG is Lanczos in disguise.* Forward reference to 26, then the
  identification made explicit once 26 is done. *What finite precision does:* in exact
  arithmetic CG terminates in $n$ steps, in floating point it does not, and orthogonality among
  the residuals is lost in exactly the way Lanczos loses it.
- **New experiments:** measure $\|e_k\|_A$ against the $\sqrt{\kappa}$ bound for several
  condition numbers; run CG past $n$ steps and show it does not terminate; measure residual
  orthogonality decaying
- **Prerequisites:** 15, 16, 20, 23
- **Later dependencies:** 25, 26, 87

### Lesson 25, Preconditioning (PLANNED, strengthen)

- **T&B concepts to integrate:** L40
- **Missing:** left, right and split preconditioning as distinct operations; clustering as the
  real goal
- **New sections required:** *Three ways to precondition* and why the choice matters for what
  the residual means. *Clustering beats conditioning:* a spectrum in two tight clusters
  converges in about two iterations regardless of $\kappa$, which the $\sqrt\kappa$ bound
  completely fails to predict.
- **New experiments:** build a matrix with clustered eigenvalues and a large $\kappa$, run CG,
  and show it beats the bound by orders of magnitude. Then the cost trade-off: total time
  against preconditioner strength, showing the U shape.
- **Prerequisites:** 24
- **Later dependencies:** 27, 28

### Lesson 26, Krylov Subspaces, Arnoldi and Lanczos (PLANNED, strengthen)

- **T&B concepts to integrate:** L33, L36
- **Missing:** the Arnoldi relation stated and verified; ghost eigenvalues; reorthogonalization
  strategies
- **New sections required:**
  1. *The Arnoldi relation* $AQ_m = Q_{m+1}\tilde{H}_m$, derived and then verified numerically
     to machine precision. Everything downstream is a consequence of this identity.
  2. *Ghost eigenvalues.* Once Lanczos loses orthogonality, converged Ritz values reappear as
     spurious duplicates. Show them.
  3. *Reorthogonalization:* none, full, and selective, with the cost against accuracy
     trade-off.
- **New experiments:** verify the Arnoldi relation residual is $O(u)$; measure $\|Q^*Q - I\|$
  growing in Lanczos and staying flat in Arnoldi with full reorthogonalization; produce ghost
  eigenvalues and remove them
- **Prerequisites:** 15, 16, 23, 24
- **Later dependencies:** 27, 39, 65

### Lesson 27, GMRES and Nonsymmetric Solvers (PLANNED, strengthen)

- **T&B concepts to integrate:** L35, **L39 (currently a mention only)**
- **Missing:** stagnation; nonnormality breaking eigenvalue-based prediction; a real
  biorthogonalization treatment
- **New sections required:**
  1. *Stagnation.* GMRES can make no progress at all for $n-1$ steps and then solve exactly.
     Construct such a matrix.
  2. *Eigenvalues do not determine GMRES convergence.* For a nonnormal matrix, any convergence
     curve is possible with any spectrum. This is a striking T&B result.
  3. *Biorthogonalization and BiCG.* Two-sided Lanczos, biorthogonality, the short recurrence,
     and breakdown as the price paid for it.
- **New experiments:** GMRES stagnating on a shift matrix; residual curves for GMRES against
  BiCGSTAB on the same nonsymmetric system; memory growth of full GMRES against restarted
- **Prerequisites:** 26
- **Later dependencies:** 83

### Lesson 29, Least Squares and the Normal Equations (PLANNED, strengthen)

- **T&B concepts to integrate:** L11
- **Missing:** the projector formulation
- **New sections required:** *Least squares is a projection.* $Ax$ is the orthogonal projection
  of $b$ onto $\mathrm{range}(A)$, the projector is $P = A(A^*A)^{-1}A^*$, and the normal
  equations are exactly the statement that the residual is orthogonal to the range. Uses lesson
  16 directly.
- **Prerequisites:** **16**, 15, 19
- **Later dependencies:** 30, 31, 32, 33, 34

### Lesson 30, Gram-Schmidt and QR (PLANNED, strengthen)

- **T&B concepts to integrate:** L7, L8
- **Too shallow:** the loss of orthogonality is demonstrated but not explained
- **New sections required:**
  1. *QR through projectors.* Gram-Schmidt is triangular orthogonalization: it applies a
     sequence of projectors. Householder in lesson 31 is orthogonal triangularization, the
     mirror image. Stating this duality makes the two algorithms one idea.
  2. *Why classical Gram-Schmidt fails.* CGS subtracts projections of the **original** vector,
     MGS subtracts them from the **running** vector. In exact arithmetic these are identical.
     In floating point, MGS uses already-orthogonalized quantities and CGS does not.
- **New experiments:** $\|Q^*Q - I\|$ against $\kappa(A)$ over several decades, with a fitted
  slope. CGS should measure slope near 2 and MGS near 1. That fitted exponent is the
  quantitative statement of the difference.
- **Prerequisites:** **16**, 29
- **Later dependencies:** 31, 37

### Lesson 31, Householder and Givens QR (PLANNED, strengthen)

- **T&B concepts to integrate:** L10, L16, and the detailed analysis of L15
- **Too shallow:** the plan names the sign trick and a stability comparison without deriving
  either
- **New sections required:**
  1. *Constructing a reflector,* with the geometry of reflecting $x$ onto $\pm\|x\|e_1$.
  2. *The sign choice.* Choosing $v = x + \mathrm{sign}(x_1)\|x\|e_1$ avoids cancellation. This
     is lesson 05 applied directly, and getting it wrong is a real and common bug.
  3. *Backward stability of Householder QR,* stated and then verified.
  4. *Never form Q.* Store the reflectors and apply them, which is cheaper and more accurate.
- **New experiments:** the striking one. On an ill-conditioned matrix, measure both
  $\|\tilde{Q}\tilde{R} - A\|$ and $\|\tilde{Q}^*\tilde{Q} - I\|$ for Householder. The first
  stays at machine precision while the second may not be tiny. **Backward stability does not
  require the computed factors to be individually accurate, only their product.** Compare with
  CGS, where the product is also poor.
- **Prerequisites:** 30
- **Later dependencies:** 32, 37, 38, 42

### Lesson 32, Solving Least Squares in Practice (PLANNED, strengthen substantially)

- **T&B concepts to integrate:** L18, L19
- **Too shallow:** currently one comparison; T&B gives this two full lectures
- **New sections required:**
  1. *The four condition numbers of least squares.* Sensitivity of $x$ and of $y = Ax$ to
     perturbations in $b$ and in $A$, and the angle $\theta$ between $b$ and
     $\mathrm{range}(A)$. The familiar $\kappa^2$ appears as one case, not as the whole story.
  2. *Why normal equations are unstable.* Forming $A^*A$ squares the condition number before
     any solving happens. This is the algorithm's fault, not the problem's: the problem's own
     condition number is $\kappa$, not $\kappa^2$, when the residual is small.
  3. *Which algorithm to use,* as a decision table.
- **New experiments:** solve the same rank-deficient and near-rank-deficient problem by normal
  equations, CGS, MGS, Householder QR and SVD, against a high precision reference, sweeping
  $\kappa$ from $10^2$ to $10^{14}$. Normal equations should fail at $\kappa \approx u^{-1/2}$
  and the others should survive to $\kappa \approx u^{-1}$. That crossover point is the
  quantitative statement of "squaring the condition number".
- **Prerequisites:** 29, 30, 31, 41
- **Later dependencies:** 33, 43, 92

### Lesson 35, Eigenvalue Theory and Localization (PLANNED, strengthen)

- **T&B concepts to integrate:** L24, L25
- **Missing:** why eigenvalue computation must be iterative
- **New sections required:** *No finite algorithm exists.* Eigenvalues are roots of the
  characteristic polynomial, every polynomial is the characteristic polynomial of its companion
  matrix, and by Abel and Galois no formula in radicals exists for degree 5 or more. Therefore
  any eigenvalue algorithm must be iterative. This single argument explains why Part 6 is
  organised completely differently from Part 3, and it belongs at the start of Part 6.
  Also: *a roadmap* of the algorithms to come and what each is for.
- **Prerequisites:** 15, 19
- **Later dependencies:** all of Part 6

### Lesson 36, Power Methods (PLANNED, strengthen)

- **T&B concepts to integrate:** L27
- **Missing:** why the Rayleigh quotient is second order accurate
- **New sections required:** *The Rayleigh quotient is a stationary point.* $\rho(x)$ has zero
  gradient at an eigenvector, so an $O(\epsilon)$ error in the vector gives an $O(\epsilon^2)$
  error in the eigenvalue. That is what makes RQI cubic for symmetric matrices rather than
  quadratic, and it is currently unexplained.
- **New experiments:** measure eigenvector error and eigenvalue error separately on the same
  run and confirm the eigenvalue error is the square of the vector error. Measure the observed
  order of RQI as 3 using `nalib.convergence`.
- **Prerequisites:** 35
- **Later dependencies:** 37, 39

### Lesson 39, Krylov Methods for Eigenvalues (PLANNED, strengthen)

- **T&B concepts to integrate:** L34
- **Missing:** Rayleigh-Ritz stated as a projection method
- **New sections required:** *Rayleigh-Ritz.* Project the eigenvalue problem onto the Krylov
  subspace, solve the small problem, lift the answer back. This is lesson 16 reappearing, and
  saying so explicitly is what connects Part 4 to Part 6. Also which Ritz values converge
  first, and why extremal eigenvalues come first.
- **New experiments:** plot Ritz values against iteration and watch the outermost converge
  first; use the Ritz residual as a stopping test and confirm it bounds the true error
- **Prerequisites:** 26, 35, 36
- **Later dependencies:** 96

### Lesson 41, SVD Theory (PLANNED, strengthen)

- **T&B concepts to integrate:** L4
- **Missing:** conditioning through singular values, derived
- **New sections required:** derive $\|A\|_2 = \sigma_1$, $\|A^{-1}\|_2 = 1/\sigma_n$ and hence
  $\kappa_2(A) = \sigma_1/\sigma_n$. This closes the loop opened in lesson 15, where the induced
  2-norm could only be forward referenced.
- **Prerequisites:** 16, 35
- **Later dependencies:** 42, 43, 32

### Lesson 43, SVD Applications and Low Rank (PLANNED, strengthen)

- **T&B concepts to integrate:** L5
- **Missing:** singular value perturbation
- **New sections required:** the Weyl inequality $|\sigma_i(A+E) - \sigma_i(A)| \le \|E\|_2$.
  Singular values are perfectly conditioned, which is exactly why the SVD is the right tool for
  deciding numerical rank, and why eigenvalues of a nonsymmetric matrix are not.
- **Prerequisites:** 41
- **Later dependencies:** 92

### Lesson 65, Gaussian Quadrature (PLANNED, promote from supplementary to core)

- **T&B concepts to integrate:** L37
- **Currently planned:** Golub-Welsch listed as supplementary
- **New sections required:** *Gauss quadrature is an eigenvalue problem.* The three-term
  recurrence of the orthogonal polynomials assembles into a symmetric tridiagonal Jacobi
  matrix. Its eigenvalues are the Gauss nodes, and the squared first components of its unit
  eigenvectors are the weights. The same tridiagonal matrix is what Lanczos produces, so
  computing quadrature rules and computing eigenvalues of a sparse matrix are the same
  computation.
- **New experiments:** implement Golub-Welsch and verify the nodes and weights against
  `numpy.polynomial.legendre.leggauss` to machine precision; confirm degree of precision
  $2n-1$ exactly
- **Prerequisites:** 26, 38, 55 (orthogonal polynomials)
- **Later dependencies:** none

---

## 7. Advanced consequences that must be added

The brief asks for the mathematical evolution, not isolated algorithms. These are the chains
the course must make explicit, with the lesson where each link is forged.

**Chain 1, orthogonality to stable algorithms**

```
orthogonal vectors (16) -> orthonormal bases (16) -> projectors (16)
   -> Gram-Schmidt (30) -> QR (30) -> Householder (31) -> least squares (29, 32)
   -> stable algorithms (31, 32)
```
Limitation exposed at each step: Gram-Schmidt loses orthogonality, so Householder was
developed. Normal equations square the condition number, so QR-based least squares was
developed.

**Chain 2, floating point to reliable computation**

Already complete and verified in Part 1, lessons 03 to 06. The chain the brief names maps
exactly onto lessons 03, 04, 05, 06.

**Chain 3, the eigenvalue evolution**

```
eigenvalue problem (35) -> no finite algorithm exists (35) -> power method (36)
   -> too slow, only the dominant pair -> inverse iteration (36)
   -> needs a good shift -> shifted inverse iteration (36)
   -> where does the shift come from -> Rayleigh quotient iteration (36)
   -> only one eigenvalue at a time -> simultaneous iteration (37)
   -> too expensive -> QR algorithm (37) -> too slow per step -> Hessenberg first (37)
   -> still linear -> shifted QR (37) -> Schur form (37)
   -> matrix too large to store -> Krylov eigensolvers (39)
```
Every arrow is a **limitation and its cure**, and the course must state each arrow, not just
the boxes.

**Chain 4, the Krylov evolution**

```
Krylov subspace (26) -> naive basis is hopelessly ill conditioned (26)
   -> orthogonalize it -> Arnoldi (26) -> Arnoldi relation (26)
   -> symmetric case gives a 3-term recurrence -> Lanczos (26)
   -> but orthogonality is lost -> ghost eigenvalues (26) -> reorthogonalization (26)
   -> project the eigenproblem -> Ritz values (39)
   -> project the linear system, minimize the residual -> GMRES (27)
   -> memory grows without bound -> restarted GMRES (27)
   -> symmetric positive definite case -> CG (24), which is Lanczos (26)
   -> nonsymmetric but want a short recurrence -> biorthogonalization, BiCG (27)
   -> which can break down: the price of the short recurrence (27)
   -> convergence too slow in all cases -> preconditioning (25)
   -> the tridiagonal matrix is also a Jacobi matrix -> Gauss quadrature (65)
```

**Chain 5, the SVD**

```
SVD (41) -> singular values (41) -> rank (41) -> numerical rank (43)
   -> four fundamental subspaces (41) -> pseudoinverse (41)
   -> least squares, including rank deficient (32) -> conditioning (41)
   -> low rank approximation and Eckart-Young (43)
   -> too expensive at scale -> randomized SVD (43, modern supplement)
```

**Chain 6, direct to iterative**

```
Gaussian elimination (17) -> O(n^3) and O(n^2) memory (08, 17)
   -> impossible at n = 10^6 -> exploit sparsity (21)
   -> fill-in destroys sparsity (21) -> stop factorizing, iterate (23)
   -> stationary methods converge too slowly (23) -> Krylov (24, 26, 27)
   -> still slow when ill conditioned -> preconditioning (25)
   -> want iteration count independent of n -> multigrid (28)
```

---

## 8. The one new lesson

### Lesson 16, Orthogonality and Projectors

**Position:** Part 3, Direct Methods for Linear Systems, immediately after lesson 15 (norms).

**Justification against the four tests in the brief:**

1. **Is the concept important?** It is the hinge of Trefethen and Bau's entire Part II, and
   three separate parts of this course depend on it: Part 4 (Rayleigh-Ritz is a projection),
   Part 5 (least squares is a projection, QR builds the projector), and Part 6 (the SVD is an
   orthogonal change of basis, and Krylov eigensolvers project).
2. **Can it fit into an existing lesson?** No good option exists. Folding it into lesson 15
   would make one lesson carry norms, orthogonality and projectors. Folding it into lesson 30
   (Gram-Schmidt) would place it **after** Part 4, which already needs it.
3. **Would combining overload the host lesson?** Yes. Lesson 15 is already the norms lesson and
   is a prerequisite for everything that follows.
4. **Does it deserve standalone treatment?** Yes. It has its own theorems, its own geometry and
   its own experiments, and it is referenced from at least six later lessons.

**Why this position and not later.** Part 4 (Krylov) comes before Part 5 (QR) in this course, a
deliberate choice recorded in `COURSE_ARCHITECTURE.md` section 3. Arnoldi and Rayleigh-Ritz are
projection methods, so projectors must be available before Part 4 begins. Placing lesson 16 at
the end of the norms material satisfies that and costs nothing, because the two ideas are
neighbours: norms measure length, orthogonal matrices preserve it.

**Planned content:**

| Section | Content |
|---|---|
| 1 | Inner products, orthogonality, orthonormal sets, expansion of a vector in an orthonormal basis |
| 2 | Orthogonal and unitary matrices, and the key fact $\|Qx\|_2 = \|x\|_2$ |
| 3 | Why that fact matters: an orthogonal transformation cannot amplify error, so $\kappa_2(Q) = 1$. This is why every stable algorithm in the course is built from reflections and rotations. |
| 4 | Orthogonal changes of coordinates, and the geometry of rotation and reflection |
| 5 | Projectors: $P^2 = P$, complementary projectors $I - P$, oblique versus orthogonal |
| 6 | Orthogonal projectors: $P^* = P$, and $P = QQ^*$ for orthonormal $Q$ |
| 7 | Rank-one projectors, $P = qq^*$ and $I - qq^*$, which are the building blocks of Gram-Schmidt |
| 8 | Projection error, and the orthogonality of the residual to the subspace |
| 9 | Forward look: this is what QR computes (30), what least squares means (29), and what Rayleigh-Ritz does (39) |

**Experiments:**

1. Verify $\kappa_2(Q) = 1$ for random orthogonal $Q$, and confirm $\|Qx\|_2 = \|x\|_2$ to
   machine precision. Contrast with a random non-orthogonal matrix of the same size, where the
   norm changes by orders of magnitude.
2. Verify $P^2 = P$ and $P^* = P$ numerically for $P = QQ^*$, and confirm the residual
   $b - Pb$ is orthogonal to every column of $Q$.
3. Show that an oblique projector can have arbitrarily large norm while an orthogonal projector
   always has norm exactly 1. This is the quantitative reason orthogonal projection is the
   numerically safe kind.

**Prerequisites:** 06 (conditioning and stability), 15 (norms)

**Unlocks:** 24, 26, 27, 29, 30, 31, 32, 39, 41

**Library additions:** `nalib/orthogonality.py` with projector construction, orthogonality
measures, and random orthogonal matrix generation for testing.

---

## 9. Numerical experiments that must be added

Each answers a specific question, per the brief's standard.

| # | Question it answers | Lesson | Measure | Expected result |
|---|---|---|---|---|
| 1 | Can an orthogonal matrix amplify error? | 16 | $\kappa_2(Q)$, $\|Qx\|_2/\|x\|_2$ | exactly 1, to machine precision |
| 2 | How bad can an oblique projector be? | 16 | $\|P\|_2$ as the angle closes | unbounded, while orthogonal stays at 1 |
| 3 | How accurately can a badly conditioned triangular system be solved? | 17 | forward error against $\kappa u$ | far better than $\kappa u$, because the componentwise bound is stronger |
| 4 | Does pivot growth actually reach its worst case? | 18 | growth factor on Wilkinson's matrix, then on 10000 random matrices | $2^{n-1}$ on the constructed one, about $n^{2/3}$ on random ones |
| 5 | Why exactly is classical Gram-Schmidt worse than modified? | 30 | $\|Q^*Q - I\|$ against $\kappa(A)$, fitted slope | CGS slope near 2, MGS near 1 |
| 6 | Can an algorithm be backward stable while producing an inaccurate $Q$? | 31 | $\|\tilde{Q}\tilde{R} - A\|$ and $\|\tilde{Q}^*\tilde{Q} - I\|$ separately | product stays at $O(u)$ even when orthogonality degrades |
| 7 | Where exactly do normal equations fail? | 32 | forward error against $\kappa$, five methods | normal equations break at $\kappa \approx u^{-1/2} \approx 10^8$, the rest survive to $10^{16}$ |
| 8 | Is the eigenvalue error really the square of the eigenvector error? | 36 | both, per iteration | yes, confirming the Rayleigh quotient is stationary |
| 9 | Does the Arnoldi relation hold in floating point? | 26 | $\|AQ_m - Q_{m+1}\tilde{H}_m\|$ | $O(u)$, even when orthogonality is lost |
| 10 | What do ghost eigenvalues look like? | 26 | Ritz values against iteration, no reorthogonalization | converged values reappearing as spurious duplicates |
| 11 | Do eigenvalues predict GMRES convergence? | 27 | residual curves for matrices with identical spectra but different nonnormality | no, the curves differ completely |
| 12 | Does clustering beat conditioning? | 25 | CG iterations for clustered versus spread spectra at equal $\kappa$ | clustered converges in about the number of clusters, ignoring $\kappa$ |
| 13 | Does CG actually terminate in $n$ steps? | 24 | run past $n$, measure residual orthogonality | no, and the loss of orthogonality is the reason |
| 14 | Are Gauss nodes really eigenvalues? | 65 | Golub-Welsch against `leggauss` | agreement to machine precision |

---

## 10. Stability and error analysis that must be added

The brief insists these are kept distinct: is the **problem** ill conditioned, and separately,
is the **algorithm** stable.

| Lesson | Conditioning question | Stability question | Both must appear |
|---|---|---|---|
| 17 | conditioning of a triangular system, and why the normwise $\kappa$ overstates it | backward stability of back substitution, componentwise | yes |
| 18 | unchanged from 19 | backward stability of GE with partial pivoting, and the growth factor as the only thing that can go wrong | yes |
| 19 | $\kappa(A)$ derived, perturbations in $A$ and $b$ separately | reference back to 17 and 18 | yes |
| 20 | conditioning of an SPD system | Cholesky needs no pivoting, and why | yes |
| 30 | conditioning of the basis | CGS is unstable, MGS is conditionally stable | yes |
| 31 | unchanged | Householder QR is backward stable, stated and verified | yes |
| 32 | the four least squares condition numbers | normal equations unstable, QR and SVD stable | yes, and this pairing is the whole lesson |
| 36 | eigenvalue conditioning for normal and nonnormal matrices | stability of the power iteration under normalization | yes |
| 37 | unchanged | backward stability of the QR algorithm, and why orthogonal similarity is the reason | yes |
| 41 | singular values are perfectly conditioned | why forming $A^*A$ is unstable | yes |
| 26 | conditioning of the Krylov basis, which is terrible before orthogonalization | loss of orthogonality in Lanczos | yes |

**Standing rule for these lessons:** never write a generic stability paragraph. State the
backward error bound, then measure it.

---

## 11. Dependency changes

**New dependency edges created by lesson 16:**

```
15 norms ---> 16 orthogonality and projectors
06 conditioning ---> 16

16 ---> 24 conjugate gradient          (A-orthogonality is orthogonality in another inner product)
16 ---> 26 Arnoldi and Lanczos         (orthogonalizing the Krylov basis)
16 ---> 29 least squares               (least squares IS a projection)
16 ---> 30 Gram-Schmidt and QR         (Gram-Schmidt applies projectors)
16 ---> 31 Householder                 (reflectors are orthogonal transformations)
16 ---> 39 Krylov eigensolvers         (Rayleigh-Ritz is a projection method)
16 ---> 41 SVD theory                  (the SVD is an orthogonal change of basis)
```

**Strengthened existing edges:**

```
26 Arnoldi/Lanczos ---> 65 Gauss quadrature
      new, and substantial. The Lanczos tridiagonal matrix is the Jacobi matrix.

24 CG <---> 26 Lanczos
      new, bidirectional. They are the same recurrence.

41 SVD theory ---> 19 conditioning of linear systems
      backward reference: kappa_2 = sigma_1/sigma_n is derived in 41 and used in 19.
      Lesson 19 forward references it rather than asserting it.
```

**Documents to regenerate:** `COURSE_MAP.md` (generated), `COURSE_DEPENDENCIES.md` (hand
written, needs the lesson 16 edges and the renumbering),
`LEARNING_PATH.md` (hand written, all route listings shift by one above lesson 15),
`README.md` (part ranges), `COURSE_ARCHITECTURE.md` (part table and phase table).

---

## 12. Final coverage status

Every lecture in the reference document has exactly one status. No entry is unresolved.

| Status | Count | Lectures |
|---|---|---|
| **FULLY COVERED** by an existing lesson plan | 11 | L9, L13, L20, L21, L23, L24, L26, L28, L29, L30, L31 |
| **INTEGRATED** into an existing lesson, named piece added | 16 | L1, L3, L4, L5, L7, L11, L12, L22, L25, L27, L32, L33, L35, L36, L38, L40 |
| **WILL BE STRENGTHENED**, substantial new sections and experiments | 11 | L8, L10, L14, L15, L16, L17, L18, L19, L34, L37, L39 |
| **REQUIRES NEW LESSON** | 2 | L2, L6, both into new lesson 16 |
| Unresolved | **0** | none |

Totals: 11 + 16 + 11 + 2 = **40 lectures, all accounted for.**

### Modern extensions, classified

The brief asks separately about material beyond the book. None of the following is presented as
Trefethen and Bau content.

| Topic | Classification | Lesson | Already planned |
|---|---|---|---|
| Sparse storage, fill-in, reordering | CORE | 21 | yes |
| Matrix-free operators | CORE | 26 | to be added, since Krylov methods need only a matvec, and saying so explicitly is the point of the method |
| Restarted Krylov, implicit restarting | ADVANCED | 27, 39 | yes |
| Multigrid | ADVANCED | 28 | yes |
| Rank-revealing QR with column pivoting | ADVANCED | 33 | yes |
| Total least squares | ADVANCED | 33 | yes |
| Truncated SVD and regularization | ADVANCED | 32, 43 | yes |
| Randomized SVD, randomized range finder | MODERN SUPPLEMENT | 43 | yes |
| Sketching for least squares | MODERN SUPPLEMENT | 43 | to be added as a short section |
| Inverse problems, Tikhonov, the L-curve | ADVANCED | 32, 96 | yes |
| Low-rank adapters, embeddings | MODERN SUPPLEMENT | 92 | yes |
| Krylov methods at scale in learning | MODERN SUPPLEMENT | 96 | yes |

Two additions to the existing plan: **matrix-free operators** as a section in lesson 26, and
**sketching** as a short section in lesson 43.

---

## What happens next

No lesson file has been modified. This document is the plan.

The structural change (one inserted lesson, renumbering above it) touches only planning
metadata and generated documents, because all eight written lessons sit below the insertion
point. It is recorded in [`CURRICULUM_CHANGE_LOG.md`](CURRICULUM_CHANGE_LOG.md).
