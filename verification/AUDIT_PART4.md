# Audit: Part 4, Iterative and Krylov Subspace Methods

A part-level audit. [FINAL_AUDIT.md](FINAL_AUDIT.md) carries the current overall position and
links to every part-level audit, so that list is maintained in one place rather than repeated
here and going stale each time a part lands.

Everything below is measured from the repository by [`run_all.py`](run_all.py). No number in
this document was typed in by hand.

---

## Verdict

**Part 4 passes.** Six lessons, all executed, all tested, all numerically verified, no
placeholders, no unpaired files, no hardcoded dimensions.

| Check | Result |
|---|---|
| Notebooks executed end to end | **6 of 6 pass** (28 of 28 across the repository) |
| Automated tests | **2312 of 2312 pass** (1146 new) |
| Assertions inside the Part 4 lessons | **42, all passing** |
| Concepts covered from the sources | **32 new, 179 of 507 total** |
| Concepts outstanding in Parts 1 to 4 | **0** |
| Worked exercise solutions | **102**, one for every exercise |
| Generality scan | **0 candidates across 132 files** |

---

## A. Curriculum completeness

Part 4 was planned as 6 lessons. Every one exists.

| Lesson | Title | Figures | Assertions | Words |
|---|---|---:|---:|---:|
| 23 | Classical Iterative Methods | 2 | 5 | 5630 |
| 24 | Conjugate Gradient | 2 | 8 | 5329 |
| 25 | Preconditioning | 1 | 7 | 3919 |
| 26 | Krylov Subspaces, Arnoldi and Lanczos | 2 | 7 | 4507 |
| 27 | GMRES and Nonsymmetric Solvers | 2 | 12 | 6066 |
| 28 | Multigrid Methods | 2 | 3 | 5897 |

**Nothing planned was dropped.**

## B. Source completeness

| Source | Concepts | Status |
|---|---:|---|
| Trefethen and Bau, L32 to L40 | 17 | **complete** |
| Sauer, 2.5 and 4.4 | 8 | **complete** |
| Gupta, 5 (iterative sections) | 5 | **complete** |
| Supplementary (multigrid) | 2 | **complete** |

Trefethen and Bau supplies more than half of Part 4's concepts, which is the pattern the
original instruction asked for: the numerical linear algebra parts lean on it hardest.

## C. Numerical linear algebra depth

| Requested depth | Where, and what is measured |
|---|---|
| Krylov spaces as what matrix-vector products can reach | Lesson 26 sections 1 and 2, with a $50{,}000$ unknown solve and **no matrix stored** |
| The collapse of the naive basis | Lesson 26, condition number growing to $1.7\times10^{41}$ by $m = 30$ |
| Arnoldi and Lanczos, and why symmetry gives a short recurrence | Lesson 26 sections 3 and 4, relation verified to roundoff at every size |
| Loss of orthogonality and ghost eigenvalues | Lesson 26 sections 7 and 8, and Paige's proportionality confirmed in the solutions to within a factor of 1.7 over fourteen orders of magnitude |
| CG as optimality over a Krylov space | Lesson 24, with the $O(\sqrt{\kappa})$ exponent fitted at **0.4843** and the constant at 9.17 against the theoretical 9.557 |
| Clustering beating $\kappa$ | Lesson 24 section 6: identical $\kappa = 10^4$, **200 iterations against 14** |
| Preconditioning as spectral surgery | Lesson 25 section 4, exponents fitted per preconditioner |
| GMRES, Givens rotations, restarting | Lesson 27 sections 2 to 4, residual verified exact to $8.4\times10^{-14}$ and free |
| Why eigenvalues do not determine GMRES convergence | Lesson 27 section 5, matrices with **identical** spectra and completely different curves |
| Biorthogonalization, BiCG and its breakdown | Lesson 27 sections 7 and 8, breakdown triggered deliberately and detected |
| Multigrid as the escape from $O(n^\alpha)$ | Lesson 28, two-grid factor **exactly $5/81$** at every $n$ to twelve digits |

## D. Computational completeness

Four library modules added, 70 public functions, 2077 lines:

| Module | Contents |
|---|---|
| `iterative` | Jacobi, damped Jacobi, Gauss-Seidel, SOR, splittings, iteration matrices, optimal $\omega$ |
| `krylov` | matrix-free operators, Arnoldi, Lanczos, Ritz values, steepest descent, CG, three preconditioners |
| `nonsymmetric` | GMRES with Givens rotations, restarting, MINRES, BiCG, BiCGSTAB, field of values, non-normality measures |
| `multigrid` | smoothers and smoothing factors, grid transfers, Galerkin coarsening, V, W and two-grid cycles, an $O(n)$ stencil hierarchy |

**Every algorithm is checked against its defining identity, not only against a reference:**

| From scratch | Checked against |
|---|---|
| `arnoldi` | $AQ_m = Q_{m+1}H_m$ directly, at every size and on four matrix families |
| `lanczos` | the Rayleigh quotient $Q^TAQ$, and the orthogonality it is known to lose |
| `conjugate_gradient` | conjugacy of the directions, finite termination, and its own convergence bound |
| `gmres` | the best residual reachable in $K_m$, computed independently by least squares |
| `minres` | GMRES on the same symmetric matrix, agreeing to $10^{-5}$ with three vectors instead of $n$ |
| `givens` | $c^2+s^2 = 1$ across the whole exponent range, where the naive formula returns `inf` |
| `two_grid_operator` | the closed form $5/81$, matched to $10^{-16}$ |
| `galerkin_coarse` | the coarse rediscretisation, **exactly equal** for the model problem |
| `smoothing_factor` | the analytic optimum $\omega = 2/3$, $\mu = 1/3$ |

**The Part 3 test gaps were closed at the same time.** `cholesky`, `banded` and `refinement`
had no dedicated test modules; they now have 447 tests between them, and writing them found
three real defects (below).

## E. Defects this process caught

Every one is a real bug or a wrong claim found by running things.

| What | How it was caught |
|---|---|
| **Lesson 23's frequency claim was backwards.** Written as "Jacobi kills high frequencies fast", when $\lambda_k = \cos(k\pi h) \to -1$ so it barely touches them | The figure disagreed with the text. Measured: plain Jacobi's worst upper-half shrinkage is **0.9988**, damped Jacobi's is **0.3333**. Section 6 rewritten, and lesson 28 now rests on the corrected fact |
| **Multigrid with plain Jacobi does not converge at any size** | The corrected fact, tested. This is the sharpest possible confirmation that lesson 23's error mattered |
| **GMRES stopped after one step, always.** The lucky-breakdown test read the subdiagonal entry **after** the Givens rotation had zeroed it | Correctness sweep across sizes. The entry is now saved before the rotation |
| **BiCG never reported a breakdown.** The test used an absolute floor of $10^{-300}$, but a numerically zero inner product is about $10^{-16}$ | Forcing a breakdown with an orthogonal shadow vector. The test is now relative, and the forced case is a test |
| **`thomas` silently zero-padded a short off-diagonal**, solving a different system and returning a confident wrong answer | A length-mismatch test. Now rejects |
| **`tridiagonal_matrix` could not infer $n = 1$** | Sweeping sizes from 1 upward |
| **`cholesky_solve` raised `IndexError` from deep inside** on a length mismatch instead of validating | A rejection test. Guards added to `cholesky_solve`, `forward_substitution` and `back_substitution` |
| **`iterations_needed` silently returned 0 for a negative rate** | An input sweep. Now rejects, since $\rho$ is a modulus |
| **The 2D V-cycle did not converge**, factors 0.86 to 0.96 | The stencil was written without the $1/h^2$ factor, so the coarse operator was four times too small. A cycle comparing residuals across grids has no freedom about scaling |
| **`prescribed_gmres_matrix` produced eigenvalues off by 0.91 at $n = 10$** | Checking the construction against its own specification. The companion matrix of $\prod(t-i)$ is hopelessly ill conditioned. Replaced by a family whose eigenvalues are its diagonal and therefore checkable |
| **Deflated restarting made things worse the more it deflated** | It carried ordinary Ritz vectors, which converge to the **largest** eigenvalues. With harmonic Ritz vectors: 3175 steps become 100 |
| **CGS converged nowhere** | The recurrence was missing its auxiliary vector `q`. Corrected, it confirms the theory: the CGS residual peak is **2.06 times the square** of BiCG's, median over seven runs |
| **`fsum` was asserted to beat working precision** for iterative refinement | It does, but only **49 times out of 72**. The exact residual wins 72 of 72. The test now asserts the pattern, not a single comparison |
| **Lesson 28 took 144 seconds to execute** | Profiling per cell. The Galerkin operators were rebuilt every cycle, making the dense implementation $O(n^3)$ **per cycle**. A setup phase brought it to 10 seconds |
| **Fourteen duplicated transcript blocks in lesson 27** | The build tool already injects notebook output; hand-written copies appeared twice in the built file |
| Hardcoded dimensions: `np.zeros(20)`, `range(4)`, a literal basis size in lesson 26 | The generality scanner, run after every build |

## F. Claims corrected against measurement

Part 4's narrative was wrong in eight places before the measurements were taken. Each is
recorded because the corrected version is more interesting than the original.

| Claim as first written | What the measurement said |
|---|---|
| Chebyshev acceleration matches SOR | SOR is **exactly twice as fast**, and $\sigma^2 = \omega^\ast - 1$ to every digit at every $n$ |
| The optimal step buys a factor of 2 over Jacobi | It buys **0.1 percent**, because $D^{-1} = I/2$ already **is** the optimal fixed step for this matrix. What it buys is safety: a badly chosen fixed step diverges |
| Polak-Ribiere is more robust than Fletcher-Reeves | Identical to within two iterations, and the restart clip **never fires**, because consecutive residuals stay orthogonal even when older ones do not |
| Adaptive SOR lands within 0.1 percent of $\omega^\ast$ | It is biased **low**, 1.7716 against 1.9521, and lesson 23 measured that low is the expensive side |
| The Elman bound is loose by 3 to 1800 times | 2 to 1786, and **vacuous** in 2 of 7 cases, exactly the nonnormal ones it was reached for |
| Non-normality makes GMRES slower | Not monotone. $\kappa(V) = 5\times10^{8}$ converged in **25** steps where the normal matrix took 33 |
| Deflation fixes restart stagnation | Not the cyclic shift's. Its eigenvectors are completely delocalised, so there is no subspace to deflate |
| Multigrid is $O(n)$, so it beats a direct method | Measured against the Thomas algorithm: **a flat factor of 12 slower** at every size, both $O(n)$ |

## Statistics for Part 4

```text
Lessons written                              : 6   (23 to 28)
Words in the 6 markdown lessons              : 31348
Figures generated                            : 11
From-scratch functions inside notebooks      : 6
Library modules added                        : 4
Public library functions added               : 70
Lines of library code added                  : 2077
Test modules added                           : 7   (4 for Part 4, 3 closing Part 3 gaps)
Automated tests added                        : 1146
Assertions inside the Part 4 notebooks       : 42
Notebooks passed                             : 6 of 6
Concepts covered by Part 4                   : 32
Worked exercise solutions written            : 102
Words in the Part 4 solutions                : 33436
Concepts outstanding in Part 4               : 0
```

Repository totals after Part 4:

```text
Lessons written                              : 28 of 96
Concepts covered                             : 179 of 507  (35.3 percent)
Library modules                              : 21
Public library functions                     : 283
Lines of library code                        : 7322
Test modules                                 : 22
Automated tests                              : 2312
Lines of test code                           : 8157
Figures generated                            : 50
Assertions across all notebooks              : 269
Words across all markdown lessons            : 145652
Worked exercise solutions                    : 452
Notebooks passed                             : 28 of 28
Placeholders found                           : 0
Hardcoded dimensions found                   : 0
```

## What is deliberately not claimed

- This is not the final audit. 68 of 96 lessons remain.
- **The dense multigrid implementation is not $O(n)$**, and lesson 28 says so before measuring
  it. `StencilHierarchy` is the one that demonstrates the complexity claim, at a flat 11
  microseconds per unknown from $n = 4095$ to $n = 1{,}048{,}575$.
- **The 2D and anisotropic multigrid results live in the solutions, not the lesson.** They are
  measured, but lesson 28 itself is one dimensional.
- **The Greenbaum, Ptak and Strakos theorem is stated and not implemented in general.** Lesson
  27 uses a family that is a special case of it and whose every property is checkable from the
  matrix, and the solutions explain why the general construction was not shipped.
- Pseudospectra are pointed at, not computed. Lesson 40 owns them.
- **Algebraic multigrid is described, not implemented.** The solution to exercise 28.5.2 gives
  the coarsening rule and what to expect, and says plainly that it is a sketch.
- The BiCG and BiCGSTAB reliability figures, 23 and 19 out of 30, are for one matrix family at
  three parameter values. They show that the short-recurrence methods fail sometimes; they are
  not a general failure rate.
