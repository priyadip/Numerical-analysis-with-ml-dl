# Audit: Part 2, Nonlinear Equations and Root Finding

A part-level audit. [FINAL_AUDIT.md](FINAL_AUDIT.md) carries the current overall position and
links to every part-level audit, so that list is maintained in one place rather than repeated
here and going stale each time a part lands.

This applies the same standard the finished repository will be held to, to the second part that
is complete.

Everything below is measured from the repository by [`run_all.py`](run_all.py). No number in
this document was typed in by hand.

Generated after the run recorded in [`summary.json`](summary.json).

---

## Verdict

**Part 2 passes.** Six lessons, all executed, all tested, all numerically verified, no
placeholders, no unpaired files, no broken links.

| Check | Result |
|---|---|
| Notebooks executed end to end | **6 of 6 pass** (14 of 14 across the repository) |
| Automated tests | **390 of 390 pass** (165 of them new in Part 2) |
| Independent identity checks | **13 of 13 pass** |
| Assertions inside the Part 2 lessons | **93, all passing** |
| Notebook and markdown pairing | **0 problems** |
| Placeholder sweep | **0 found** |
| Concepts covered from the sources | **41 new, 90 of 507 total** |
| Broken internal links | **0** |
| Concepts outstanding in Parts 1 and 2 | **0** |

---

## A. Curriculum completeness

Part 2 was planned as 6 lessons covering how to solve one nonlinear equation, then many. Every
planned lesson exists.

| Lesson | Title | Cells | Time |
|---|---|---:|---:|
| 09 | [Bracketing Methods](../02_root_finding/09_bracketing_methods.md) | 28 | 4.9 s |
| 10 | [Fixed Point Iteration](../02_root_finding/10_fixed_point_iteration.md) | 20 | 5.9 s |
| 11 | [Newton and Secant Methods](../02_root_finding/11_newton_and_secant_methods.md) | 26 | 6.1 s |
| 12 | [Convergence Theory and Sensitivity](../02_root_finding/12_convergence_theory_and_sensitivity.md) | 32 | 5.4 s |
| 13 | [Polynomial Root Finding](../02_root_finding/13_polynomial_root_finding.md) | 28 | 5.0 s |
| 14 | [Nonlinear Systems of Equations](../02_root_finding/14_nonlinear_systems_of_equations.md) | 27 | 5.6 s |

**Nothing planned was dropped.** Three things were added during the work because the material
demanded them:

1. **Lesson 12 was expanded** to carry the full Wilkinson polynomial treatment plus the general
   $u^{1/m}$ multiplicity limit, rather than mentioning conditioning in passing. It turned into
   the lesson the rest of Part 2 refers back to.
2. **Damped Newton with a line search** was added to lesson 14. The source books present plain
   Newton only, and teaching a method no production code uses would have been a gap.
3. **The companion matrix** was added to lesson 13, because it is what `numpy.roots` does, and a
   lesson that benchmarks against `numpy.roots` without explaining it is incomplete.

## B. Source completeness

Part 2 exists to absorb three source chapters, and it absorbs all three completely.

| Source | Chapter | Concepts | Status |
|---|---|---:|---|
| Sauer | 1, Solving Equations | 16 of 16 | **complete** |
| Gupta | 3, Nonlinear Equations | 13 of 13 | **complete** |
| Gupta | 4, Nonlinear Systems and Polynomials | 9 of 9 | **complete** |
| Sauer | 2.7 only, multivariate Newton and Broyden | 2 of 24 | **complete**, the other 22 concepts in chapter 2 are linear systems and belong to Parts 3 and 4 |

Full section-by-section mapping in [`coverage_report.md`](coverage_report.md).

**No topic was invented as book-derived.** Two concepts in Part 2 are supplementary, meaning
they come from neither source book, and both are labelled:

| Concept | Lesson |
|---|---|
| Companion matrix approach to polynomial roots | 13 |
| Damped Newton and line search for nonlinear systems | 14 |

**Nothing in Part 2 is attributed to Trefethen and Bau**, whose book does not cover root
finding. The one place lesson 14 points forward to it is the observation that Newton's division
becomes a linear solve, and that is flagged as a forward reference to Part 3, not as source
material for this part.

## C. Numerical linear algebra depth

**Not directly applicable to Part 2.** The linear algebra material is Parts 3 to 6, lessons 15
to 43.

What Part 2 contributes to it is the one structural fact that motivates the whole of Part 3:
Newton's step in $n$ dimensions is $J\mathbf{s} = -\mathbf{F}$, so **dividing by a number becomes
solving with a matrix**. Lesson 14 states it explicitly, measures the $O(n^3)$ cost, and shows
Broyden existing entirely to avoid it. That framing is what makes Part 3 a consequence rather
than a change of subject.

## D. Computational completeness

Algorithms implemented from scratch in Part 2, each then checked against a trusted reference
and against its defining identity:

**`nalib.roots`, 15 public names**

`bisection`, `regula_falsi`, `illinois`, `fixed_point`, `newton`, `secant`, `muller`,
`chebyshev`, `halley`, `aitken`, `steffensen`, `brent`, `bisection_steps_needed`,
plus the `RootResult` history container.

**`nalib.polyroots`, 13 public names**

`descartes_sign_changes`, `descartes_bounds`, `sturm_sequence`, `sturm_sign_changes`,
`sturm_count`, `cauchy_bound`, `birge_vieta`, `bairstow`, `graeffe_step`, `graeffe_magnitudes`,
`companion_matrix`, `roots_via_companion`, `all_roots_by_deflation`.

**`nalib.nlsystems`, 8 public names**

`numerical_jacobian`, `fixed_point_system`, `seidel_system`, `newton_system`, `damped_newton`,
`broyden`, `spectral_radius`, plus the `SystemResult` container.

**No algorithm in Part 2 is "implemented" by calling a library function.** Where a library is
used it is used as the *reference to check against*, never as the implementation:

| From scratch | Checked against |
|---|---|
| `brent` | `scipy.optimize.brentq`, agreeing to 5e-15 |
| `newton_system` | `scipy.optimize.root` on a 3 by 3 system |
| `roots_via_companion` | `numpy.roots` |
| `bairstow`, `birge_vieta`, `all_roots_by_deflation` | `numpy.roots` |
| `broyden` | the secant condition $B\mathbf{s} = \mathbf{y}$ verified directly, plus a proof by construction that no other matrix satisfying it is closer to $B_k$ in Frobenius norm |
| `sturm_count` | the true root count, known exactly because the test polynomials are built from their roots |
| `numerical_jacobian` | the analytic Jacobian, to 1e-9 |

Each lesson also builds a teaching version alongside the library version, so the reader sees the
algorithm written out before importing it: `bisection_teaching`, `newton_teaching`,
`newton_system_teaching`, `broyden_teaching`, `birge_vieta_teaching`, `isolate_roots` and
`cobweb`. 17 such functions across the six notebooks.

## E. Verification completeness

### Execution

All 6 notebooks rebuilt from source and executed end to end with `nbclient`. Total 32.9 seconds.
Detail in [`notebook_execution_report.md`](notebook_execution_report.md).

### Tests

165 new tests across three modules, all passing:

| Module | Tests |
|---|---:|
| `tests/test_roots.py` | 78 |
| `tests/test_nlsystems.py` | 46 |
| `tests/test_polyroots.py` | 41 |

Plus 6 added to `tests/test_errors.py` for `diagnose_root`. Repository total 390.

### Numerical checks

93 assertions inside the Part 2 notebooks, 148 across the repository. Every convergence order
claimed in a table is measured, not asserted, and where it cannot be measured the lesson says so
and measures the work instead.

### Defects this process actually caught

These are real bugs and real wrong claims found by running things, not hypotheticals. Each was
fixed, and the fix is in the repository.

| What | How it was caught |
|---|---|
| **Brent's method was badly broken.** It reported convergence at 1.3333 with an error of 8e-2 | Comparing against `scipy.optimize.brentq`. Rewritten to the standard zeroin structure, now agreeing to 5e-15 |
| **Sturm sequences fail at a root gap of 1e-5**, earlier than the companion matrix, and lesson 13 claimed they were the one fully reliable procedure | A test asserting Sturm could resolve a 1e-8 gap. It could not. The lesson now measures the limit and states it |
| **Damped Newton took 937 iterations where plain Newton took 5** on the Rosenbrock gradient, contradicting the lesson's implication that damping only helps | Writing `test_plain_newton_fails_where_damped_newton_succeeds` and having it fail. Lesson 14 now has a section on when damping is the wrong idea |
| **Halley ranks above Newton on the efficiency index**, $3^{1/3} = 1.4422$ against $2^{1/2} = 1.4142$. The narrative said the opposite | Computing the indices instead of recalling them. Confirmed by measurement: 64 evaluations against 67 |
| **Muller's per-step order is not reliably measurable**, reporting 1.66, 2.40 and 2.61 from three starting triples | Measuring it three ways. Moved from the "measurable order" group to the "measure the work instead" group |
| **Bisection's error bound appeared violated** near the root of $x^2 - 2$ | A test at the roundoff floor. The cause is that $\sqrt2$ is not representable, so the test now restricts to brackets several ulps wide, and a second test documents the floor |
| **Birge-Vieta from $x_0 = 5$ converges to the root at 3, not the nearer root at 7** | Asserting it would find the nearest root. Turned into a teaching point: Newton gives no control over which root you get |
| **Lesson 12 defined no function and lesson 13 produced no figure**, so two coverage flags were unset | `mark_covered.py`, which reads evidence from built artifacts rather than accepting a claim. Fixed by adding `diagnose_root` and the deflation degradation plot, not by relaxing the rule |
| Several narrative claims disagreeing with the printed output: "regula falsi lost on four of six" (two of six), "two roots complex" (four of five), "the bottom of the V near 1e-8" (that was the luckiest grid point, not the typical one) | Reading the executed output against the surrounding prose, lesson by lesson |
| `f-string` syntax errors from escaped quotes inside format specifiers, in lessons 10 and 12 | Execution. The notebooks would not run at all |

The pattern worth noting: **most of these were caught by tests that were written expecting to
pass.** A test that only ever confirms what you already believe is not doing much. Six of the
ten entries above came from a test failing in a way that turned out to be correct.

### Deliberate failures

Four experiments in Part 2 fail on purpose, because the failure is the lesson: regula falsi
stagnating on a convex function, Newton cycling forever on $x^3 - 2x + 2$, deflation breaking
down at degree 10 in the wrong order, and Graeffe overflowing after 7 steps. Each is set up
explicitly, asserted to fail in the expected way, and explained.

## Statistics for Part 2

Measured from the repository, not estimated.

```text
Lessons written                              : 6   (09 to 14)
Notebooks created                            : 6
Markdown lessons created                     : 6
Words in the 6 markdown lessons              : 32883
Figures generated                            : 12
From-scratch functions inside notebooks      : 17
Library modules added                        : 3   (roots, polyroots, nlsystems)
Public library functions added               : 36
Lines of library code added                  : 1304
Test modules added                           : 3
Automated tests added                        : 165 (+6 in test_errors)
Assertions inside the Part 2 notebooks       : 93
Notebooks executed                           : 6
Notebooks passed                             : 6
Notebooks failed                             : 0
Part 2 notebook execution time               : 32.9 s
Concepts covered by Part 2                   : 41
Source chapters fully absorbed               : 3
Supplementary concepts used                  : 2
Concepts outstanding in Part 2               : 0
Worked exercise solutions written            : 97  (every exercise in Part 2)
```

Repository totals after Part 2:

```text
Lessons written                              : 14 of 96
Concepts covered                             : 90 of 507  (17.8 percent)
Library modules                              : 9
Public library functions                     : 110
Lines of library code                        : 2887
Test modules                                 : 9
Automated tests                              : 390
Lines of test code                           : 2784
Notebooks passed                             : 14 of 14
Assertions inside notebooks                  : 148
Independent identity checks                  : 13 of 13
Placeholders found                           : 0
Unpaired .ipynb or .md files                 : 0
Broken internal markdown links               : 0
```

## What is deliberately not claimed

- This is not the final audit. 82 of 96 lessons remain.
- Coverage is 17.8 percent of tracked concepts, which is Parts 1 and 2's share. No claim is
  made about the other 417 beyond that they are tracked and assigned to a named lesson.
- The efficiency measurements in lessons 11 and 12 count function evaluations, which is
  machine independent, but the **ranking** they produce depends on the four test problems
  chosen. Lesson 12 says so, and explains that hybrid methods beat their nominal index because
  these problems converge in too few steps for an asymptotic index to apply.
- The convergence orders measured for Illinois, Brent and Muller are reported as not reliably
  measurable rather than as agreeing with theory. That is a finding, and it is stated as one.
- Sturm's floating point resolution limit of about $10^{-4}$ was measured on one family of
  polynomials. The mechanism generalises; the exact number does not.

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
