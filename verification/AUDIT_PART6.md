# Audit: Part 6, Eigenvalue Problems and the Singular Value Decomposition

A part-level audit. [FINAL_AUDIT.md](FINAL_AUDIT.md) carries the current overall position and
links to every part-level audit, so that list is maintained in one place rather than repeated
here and going stale each time a part lands.

Everything below is measured from the repository by [`run_all.py`](run_all.py). No number in
this document was typed in by hand.

---

## Verdict

**Part 6 passes.** All nine lessons exist, all are executed, tested, numerically verified and
fully solved, with no placeholders, no unpaired files and no hardcoded dimensions.

| Check | Result |
|---|---|
| Lessons written | **9 of 9** |
| Notebooks executed end to end | **9 of 9 pass** (43 of 43 across the repository) |
| Automated tests | **3775 of 3775 pass** (749 new) |
| Assertions inside the Part 6 lessons | **48, all passing** |
| Concepts covered from the sources | **47 new, 261 of 507 total** |
| Concepts outstanding in Part 6 | **0** |
| Worked exercise solutions | **153**, one for every exercise |
| Generality scan | **0 candidates across 205 files** |
| Mangled escape scan | **0 control characters across the repository** |

---

## A. Curriculum completeness

Part 6 was planned as 9 lessons. **All nine exist.**

| Lesson | Title | Figures | Assertions | Words | Exercises |
|---|---|---:|---:|---:|---:|
| 35 | Eigenvalue Theory and Localization | 1 | 9 | 4138 | 17 |
| 36 | Power Methods | 1 | 6 | 4286 | 17 |
| 37 | The QR Algorithm | 1 | 4 | 3919 | 17 |
| 38 | The Symmetric Eigenvalue Problem | 1 | 6 | 3931 | 17 |
| 39 | Krylov Methods for Eigenvalues | 1 | 4 | 3492 | 17 |
| 40 | The Generalized Eigenvalue Problem | 1 | 4 | 3393 | 17 |
| 41 | SVD Theory | 1 | 5 | 3529 | 17 |
| 42 | Computing the SVD | 1 | 5 | 3500 | 17 |
| 43 | SVD Applications and Low Rank | 1 | 5 | 3042 | 17 |
| **total** | | **9** | **48** | **33230** | **153** |

**Nothing planned was dropped.**

## B. Source completeness

Part 6 covers **47 of 47** concepts attributed to it, so the part is marked complete in
[`coverage_report.md`](coverage_report.md).

Trefethen and Bau carries the greatest depth here, as the standing instruction requires. The
lectures drawn on are L24 through L34 for the eigenvalue material and L4 through L5 for the SVD,
and the specific attributions are in the generated report. Where a result is used but not proved
here, it is attributed to the source rather than presented as this course's own, and section D
below lists what is used on credit.

## C. What the measurements changed

Nine lessons produced three corrections to claims that a textbook statement or the exercise
prompt had led the first draft to expect, and one defect in the library.

**Deflation error does not accumulate with `k`.** Measured across six successive Hotelling
deflations on two spectra, the eigenvalue error is flat at 1e-15 to 1e-17 and the residual flat
at 1e-12. What grows is the cost, and only at a cluster: on a spectrum with `lambda_2/lambda_1 =
0.99` the first eigenvalue took 2285 power iterations and every later one took about 40.

**Clustered eigenvalues do not degrade the orthogonality of any method.** Jacobi, LAPACK `eigh`
and the dense Schur route all hold at 2e-15 to 6e-15 for cluster gaps from 1e-2 down to exactly
zero. What degrades is the individual eigenvector, from 1.9e-13 to 1.37, and that is correct
behaviour: inside an exact cluster no individual eigenvector is defined. The invariant subspace,
which is defined, is computed to 1.4e-15 throughout.

**The zero-shift SVD sweep is not more accurate on column-graded matrices.** Both variants reach
3e-15 out to a spread of 1e24, once the deflation criteria are correct. The genuine adversarial
structure is different and was found by search: a bidiagonal matrix with one tiny diagonal entry
among neighbours of size 1 gives the shifted sweep a relative error of **1.49e3** and the
zero-shift sweep **2.01e-13**. On every ordinary matrix tested the shifted sweep is instead 100
times more accurate and 50 to 1400 times faster.

**One library defect, found and fixed.** `nalib.svdcompute.golub_kahan_svd` tested a diagonal
entry for negligibility against the largest entry in the active block. On a graded matrix that
discards legitimate values: at spread 1e16 with the default tolerance, four of thirty diagonal
entries tripped the test and the computed singular values carried a relative error of 6.5e-3
instead of 3.3e-15. Comparing against the neighbouring off-diagonal entries instead fixes it,
leaves ordinary matrices bit for bit unchanged in both answer and sweep count, and still catches
a genuine exact zero. All 75 tests in `tests/test_svdcompute.py` pass after the change.

**One hole in this audit's own tooling, found and fixed.** `run_all.py`'s generality check
scanned `_planning/lessons_src/*.md` and `src/nalib/*.py` and reported 0 candidates while the
standalone scanner, which also reads `tests/`, reported 1: a fixed index into a singular value
spectrum in `tests/test_lowrank.py`. The check now scans the tests as well, and the test was
rewritten to compare size-independent quantities and parametrized over four image shapes.

## D. What is deliberately not claimed

- **Kaniel-Paige is not confirmed as a predictor for the restarted method.** The measured matvec
  count against the relative gap fits `gap^{-0.311}` rather than the `gap^{-0.5}` the bound
  suggests, and the count is not monotone: a gap of 1e-4 took fewer matvecs than 1e-3. The
  quantity `matvecs * sqrt(gap)` is constant to within 30 percent over the first three points and
  drops at the fourth. The bound is stated for unrestarted Lanczos from a single vector and the
  measurement is of a restarted method, which applies a polynomial filter at every restart and
  can beat it. No claim is made about which exponent is correct for the restarted case.

- **The QZ iteration is not implemented from scratch here.** The Hessenberg-triangular reduction
  is, and it is verified to 1e-15 with `H` exactly Hessenberg and `T` exactly triangular. The
  eigenvalues that follow are computed through `H T^{-1}`, which solution 40.3.1 measures failing
  exactly as `eps kappa(B)` predicts, reaching error 2.66e2 at `kappa(B) = 1e10`. The genuine
  implicit QZ bulge chase, which never forms an inverse, is described and not implemented; a
  straightforward attempt at it did not converge and was removed rather than shipped.

- **The convergence order of Rayleigh quotient iteration is not pinned down by the data.** With
  cubic convergence the iteration passes from 1e-2 to 1e-16 in two steps, so there are two or
  three points in the asymptotic regime and any fitted exponent depends on where the fit starts.
  The measured three-point exponent on the symmetric case is 2.09 and the digit counts suggest
  2.5 to 3.3. The solutions report the residual trails and the step counts, which are
  unambiguous, and say plainly that the order is between 2 and 3 by this data.

- **Block Lanczos found two copies of a doubly repeated eigenvalue from a single starting
  vector**, which exact arithmetic forbids. That is rounding acting as a block, the same
  mechanism as ghosts, and it is not reliable: at multiplicities 3 and 4 the unblocked run
  saturates at 2 while a block of 4 recovers all of them. The theoretical statement is what is
  claimed; the accidental success is reported and explicitly not depended on.

- **The Cullum-Willoughby ghost test has a range of validity, and it is measured.** At `m = 1.5n`
  it removes exactly the 20 ghosts and keeps all 40 true eigenvalues with nothing spurious left.
  At `m >= 2.5n` it deletes everything, because once the spectrum is saturated with copies every
  Ritz value is also an eigenvalue of the trimmed matrix. No claim is made that the test works
  for unbounded `m`.

- **The timing comparisons are numpy timings on one machine**, and several are dominated by
  Python interpreter overhead rather than by arithmetic. The fitted cost exponents in solutions
  38.4.1 and 42.4.3 are below their theoretical values for that reason, and the solutions say so
  rather than adjusting the numbers. Where a measured ratio is attributed to the implementation
  rather than to the algorithm, for instance one-sided Jacobi at 2551 times LAPACK, the audit
  says which and gives the figure a tuned implementation reaches instead.

- **The Gavish and Donoho threshold beat the rank sweep in one row of seven**, at noise level 1.0,
  and it did so because it can return rank 0 while the sweep searches only ranks 1 and above.
  That is a real and explainable advantage on that row, not evidence that the formula is
  generally better than an oracle sweep. On the other six rows they agree exactly or to within
  3 percent.

- **Sketched least squares with a dense Gaussian sketch is slower than solving exactly**, in
  every row of both measured problem sizes, by up to 110 times. CountSketch pays only when `m/n`
  is large, measured at 1.1x for 20000 by 50 and 3.7x for 200000 by 50. The solutions report the
  negative result rather than quoting the asymptotic advantage.

- **The single-pass randomised SVD costs a factor of 20 to 1483 against the two-pass version.**
  That is measured on this implementation, whose core recovery solves with a random projection
  of uncontrolled conditioning. Better single-pass schemes exist and are named; no claim is made
  that this factor is intrinsic to single-pass sketching.

- **Pseudospectra are computed by bisection along a ray, not as a region.** The reported
  amplification is the reach along the positive real direction from the outermost eigenvalue,
  which is one number about a set. The full boundary is not computed and no claim is made about
  the shape.

- **The departure from normality has a roundoff floor of about `sqrt(eps) ||A||_F`,** because it
  is a difference of nearly equal quantities under a square root, and this is measured rather
  than assumed: for `diag(1..8)` the true value is 0 and the computed value is 1.69e-7 against a
  predicted floor of 2.13e-7. Henrici's bound is therefore not trusted below that floor, and
  solution 35.5.3 says so where it would otherwise appear to overstate.
