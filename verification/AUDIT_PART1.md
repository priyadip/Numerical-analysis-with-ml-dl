# Audit: Part 1, Foundations of Numerical Computing

A part-level audit. [FINAL_AUDIT.md](FINAL_AUDIT.md) carries the current overall position and
links to every part-level audit, so that list is maintained in one place rather than repeated
here and going stale each time a part lands.

This is a **part-level audit**, not the final audit for the whole course. It applies the same
standard that the finished repository will be held to, to the one part that is complete.

Everything below is measured from the repository by `verification/run_all.py`. No number in
this document was typed in by hand.

Generated after the run recorded in [`summary.json`](summary.json).

---

## Verdict

**Part 1 passes.** Eight lessons, all executed, all tested, all numerically verified, no
placeholders, no unpaired files.

| Check | Result |
|---|---|
| Notebooks executed end to end | **8 of 8 pass** |
| Automated tests | **219 of 219 pass** |
| Independent identity checks | **13 of 13 pass** |
| Assertions inside the lessons | **67, all passing** |
| Notebook and markdown pairing | **0 problems** |
| Placeholder sweep | **0 found** |
| Concepts covered from the source books | **47 of 449** (Part 1's full share) |
| Broken internal links | **0** |

---

## A. Curriculum completeness

**Question: does the finished part cover what it set out to cover?**

Part 1 was planned as 8 lessons covering how numbers are represented, where error comes from,
how it propagates, when it becomes catastrophic, how to separate a hard problem from a bad
algorithm, and how to measure both accuracy and cost.

| Lesson | Planned topic | Delivered |
|---|---|---|
| 01 | Problem versus algorithm, nested evaluation, cost as a mathematical question | yes |
| 02 | Number systems, integer and fraction conversion, terminating expansions | yes |
| 03 | IEEE 754, machine epsilon, rounding, the `fl` model, subnormals, non-associativity | yes |
| 04 | Error measures, the full error taxonomy, propagation through arithmetic and functions | yes |
| 05 | Cancellation, formula rewriting, the quadratic formula, four summation algorithms | yes |
| 06 | Forward and backward error, conditioning, stability, the diagnostic procedure | yes |
| 07 | Taylor with remainder, order notation, measuring convergence order and order of accuracy | yes |
| 08 | Flop counting, BLAS levels, memory cost, measured exponents | yes |

Nothing planned for Part 1 was dropped, and two things were added during the work because the
lessons needed them: a from-scratch implementation section in lesson 02, and a figure making
Theorem 2.3 visible across bases and denominators.

## B. Source completeness

**Question: has the material from the supplied books actually been carried over, given that
the course reorganised it?**

Part 1 completely absorbs the following, with nothing outstanding from them:

- **Gupta chapters 1 and 2 in full.** Number systems, all conversions, the error taxonomy,
  round-off, overflow and underflow, machine epsilon, propagation through arithmetic and
  through functions of one and several variables, truncation error, loss of significance,
  accumulation of error.
- **Sauer chapter 0 in full.** Polynomial evaluation, binary numbers, floating point
  representation, loss of significance, the calculus review.
- **Sauer section 1.3, the forward and backward error material.** The remainder of 1.3,
  covering the Wilkinson polynomial and the sensitivity of root finding, is assigned to
  lesson 12 in Part 2.

Detail is in [`coverage_report.md`](coverage_report.md) and the concept-level table is in
[`../_planning/INTERNAL_COVERAGE_CHECKLIST.md`](../_planning/INTERNAL_COVERAGE_CHECKLIST.md).

**No topic was invented as book-derived.** Ten concepts in Part 1 are supplementary, meaning
they come from neither supplied book, and each is labelled as such in the lesson that
introduces it and listed in the coverage report.

## C. Numerical linear algebra depth

**Not applicable to Part 1.** The linear algebra material is Parts 3 to 6, lessons 15 to 42.

What Part 1 does contribute to it is the vocabulary those parts depend on: the `fl` model, the
formal definition of backward stability, the governing inequality, and the measurement tooling
for convergence order and complexity. Every factorization in Parts 3 to 6 will be checked with
the identities defined here.

The finding that **Trefethen and Bau is not in the supplied folder** is recorded in
[`../_planning/SOURCE_INVENTORY.md`](../_planning/SOURCE_INVENTORY.md) and the handling of the
depth requirement is set out in
[`../COURSE_ARCHITECTURE.md`](../COURSE_ARCHITECTURE.md), section 4.

## D. Computational completeness

**Question: is every important algorithm actually implemented, rather than called?**

| Measure | Count |
|---|---|
| Functions written from scratch inside the notebooks | 26 |
| Public functions in the reusable library `src/nalib` | 70 |
| Library modules | 6 |
| Lines of library code | 1482 |

Algorithms implemented from scratch in Part 1, each then checked against a trusted reference:

| Algorithm | Lesson | Checked against |
|---|---|---|
| Horner's rule | 01 | `numpy.polyval`, bit for bit over 2000 random cases |
| Naive polynomial evaluation | 01 | Horner, and its own derived flop count |
| Integer base conversion by repeated division | 02 | `nalib`, and Python's `format(n, 'b')` |
| Fraction base conversion by repeated multiplication | 02 | `nalib`, over 2000 random fractions |
| Machine epsilon by search | 03 | `numpy.finfo(float).eps`, exactly |
| IEEE field decomposition | 03 | exact reconstruction, over 5000 random doubles |
| Simulated low precision rounding | 03 | the `fl` bound, at 5 mantissa widths |
| Naive, pairwise, Kahan and Neumaier summation | 05 | `math.fsum`, which is exactly rounded |
| Stable quadratic formula | 05 | a 60 digit `decimal` reference |
| Forward and backward recurrences for an integral | 06 | each other, and an analytic bound |
| Bisection, fixed point, secant and Newton | 07 | their theoretical rates |
| Forward and central difference | 07 | their theoretical orders of accuracy |
| Matrix multiplication at BLAS levels 1, 2 and 3 | 08 | each other, and `A @ B` |
| Operation counter | 08 | hand-computed counts for the dot product and back substitution |

**No algorithm in Part 1 is "implemented" by calling a library function.** Where a library is
used, it is used as the thing being checked against, and that is stated.

## E. Verification completeness

**Question: were the notebooks really run, and were the numbers really checked?**

### Execution

All 8 notebooks were rebuilt from their authored sources and executed end to end by
`nbclient`. Total execution time 35.4 seconds. Zero failures. Full detail in
[`notebook_execution_report.md`](notebook_execution_report.md).

### Tests

219 tests across 6 modules, all passing. Detail in [`test_report.md`](test_report.md).

| Module | Lines |
|---|---|
| 6 test modules | 1579 lines total |

### Numerical checks

67 assertions run inside the lessons, plus 13 identity checks recomputed independently in
`verification/run_all.py` so a mistake in a lesson cannot hide itself. Detail and the
tolerance policy are in [`numerical_verification.md`](numerical_verification.md).

### Defects this process actually caught

This is the part of an audit that matters. Six real problems were found and fixed while
Part 1 was being written, and each would have survived a less demanding process.

1. **`np.longdouble` is not extended precision on this platform.** On Windows with the
   Microsoft compiler it is an alias for `float64`. Two lessons were using it as a high
   precision reference, so they were comparing values against themselves. A Sterbenz lemma
   test that could only ever pass is what exposed it. Both lessons now use `math.fsum` or
   `fractions.Fraction`. After the fix, lesson 04's measured error growth exponent stayed at
   0.53, so that conclusion survived, but it is now honestly measured.
2. **A real bug in the stable quadratic formula.** It returned the two roots in the opposite
   order from the reference, which the comparison then reported as a relative error of
   $10^{18}$. Caught by an assertion, not by inspection.
3. **An off-by-one in a derived flop count.** The naive polynomial evaluator performs $n+1$
   additions, not $n$. The measured count disagreed with the formula and the formula was
   wrong.
4. **A wrong claim about the order of magnitude spanned by a table.** Lesson 03 said 16 orders
   of magnitude where the data showed over 300.
5. **Two narrative claims that the data did not support.** Lesson 05 claimed ascending
   summation beat the given order, when the given data was already nearly sorted so the
   comparison was meaningless. Lesson 06 said two roots were complex when four were.
6. **Three test design errors.** A one-ulp perturbation cannot measure a condition number
   below 1, because the output change rounds away. The governing inequality is a first order
   statement and does not apply at a multiple root. A realised error computed in floating
   point measures the test's own rounding rather than the inequality.

### Deliberate failures

Five experiments fail on purpose, because the failure is the lesson. Each is set up
explicitly, asserted to fail in the expected way, and explained. They are listed in the
execution report so they cannot be mistaken for defects.

## Statistics for Part 1

Measured from the repository, not estimated.

```text
Source books inspected                       : 2
Source chapters inspected                    : 30  (Sauer 0-13 + 2 appendices, Gupta 1-16)
Concepts identified and tracked              : 449
Concepts covered by Part 1                   : 47  (10.5 percent)
Supplementary concepts (from neither book)   : 85  (10 used in Part 1)
Final curriculum parts                       : 14
Final curriculum lessons                     : 95
Lessons written                              : 8
Notebooks created                            : 8
Markdown lessons created                     : 8
Words in the 8 markdown lessons              : 34770
Figures generated                            : 14
From-scratch functions inside notebooks      : 26
Library modules                              : 6
Public library functions                     : 70
Lines of library code                        : 1482
Test modules                                 : 6
Automated tests                              : 219
Lines of test code                           : 1579
Notebooks executed                           : 8
Notebooks passed                             : 8
Notebooks failed                             : 0
Total notebook execution time                : 35.4 s
Assertions inside notebooks                  : 67
Independent identity checks                  : 13
Identity checks passed                       : 13
Placeholders found                           : 0
Unpaired .ipynb or .md files                 : 0
Broken internal markdown links               : 0
```

## What is deliberately not claimed

- This is not the final audit. 87 of 95 lessons remain.
- Coverage of the source books is 10.5 percent, which is exactly Part 1's share. No claim is
  made about the other 402 concepts beyond that they are tracked and assigned.
- The timing measurements in lesson 08 are machine specific. The lesson checks the *mechanism*
  behind them, showing that the exponent deficit is accounted for exactly by the growth in
  achieved Gflop/s, rather than asserting a particular number.

## How to reproduce every number in this document

```bash
pip install -r requirements.txt
python verification/run_all.py
```

That runs all seven checks: it rebuilds every notebook from source and executes it, runs the
test suite, recomputes the identity checks independently, regenerates the coverage checklist
from what actually exists, checks that every notebook has a matching markdown lesson, resolves
every relative link in every markdown file, and sweeps the whole repository for placeholders.
It writes every report in this directory and exits non-zero if anything fails.
