# Audit: Part 8, Approximation Theory and Transforms

A part-level audit. [FINAL_AUDIT.md](FINAL_AUDIT.md) carries the current overall position and
links to every part-level audit, so that list is maintained in one place rather than repeated
here and going stale each time a part lands.

Everything below is measured from the repository by [`run_all.py`](run_all.py). No number in
this document was typed in by hand.

---

## Verdict

**Part 8 passes.** All seven lessons exist, all are executed, tested, numerically verified and
fully solved, with no placeholders, no unpaired files and no hardcoded dimensions.

| Check | Result |
|---|---|
| Lessons written | **7 of 7** |
| Notebooks executed end to end | **7 of 7 pass** (60 of 60 across the repository) |
| Automated tests | **6508 of 6508 pass** (1659 new) |
| Assertions inside the Part 8 lessons | **86, all passing** |
| Concepts covered from the sources | **27 new, 347 of 507 total** |
| Concepts outstanding in Part 8 | **0** |
| Worked exercise solutions | **119**, one for every exercise |
| Generality scan | **0 candidates across 292 files** |
| Mangled escape scan | **0 control characters across the repository** |

---

## A. Curriculum completeness

Part 8 was planned as 7 lessons. **All seven exist.**

| Lesson | Title | Figures | Assertions | Words | Exercises |
|---|---|---:|---:|---:|---:|
| 54 | Function Approximation Fundamentals | 1 | 16 | 3453 | 17 |
| 55 | Orthogonal Polynomials | 1 | 11 | 3167 | 17 |
| 56 | Chebyshev Approximation and Economization | 1 | 11 | 3409 | 17 |
| 57 | Pade Rational Approximation | 1 | 6 | 3601 | 17 |
| 58 | Trigonometric Interpolation and the DFT | 1 | 15 | 3351 | 17 |
| 59 | The FFT and Signal Processing | 1 | 13 | 3507 | 17 |
| 60 | The DCT and Data Compression | 1 | 14 | 3648 | 17 |
| **total** | | **7** | **86** | **24136** | **119** |

**Nothing planned was dropped.**

Seven new library modules back the part, `approx`, `orthopoly`, `chebapprox`, `pade`, `dft`,
`fft` and `dct`, at 3563 lines, with 2453 lines of tests against them.

## B. Source completeness

Part 8 covers **27 of 27** concepts attributed to it, so the part is marked complete in
[`coverage_report.md`](coverage_report.md).

Sauer chapter 10 (Trigonometric Interpolation and the FFT) and chapter 11 (Compression) are the
main sources, and both are now complete. The approximation theory of lessons 54 and 55 is drawn
from the supplementary material rather than from a single chapter, and lessons 56 and 57 come from
Gupta chapter 11.

Nothing in this part is attributed to Trefethen and Bau that is not in it. The Bernstein ellipse
and the potential theory framing of coefficient decay are cited to it, as they were in lesson 46,
and are used rather than reproved.

## C. What the measurements changed

Seven lessons produced eight corrections to claims the first draft made, and five defects in code
written for this part.

**The DCT is not better than the DFT at energy compaction in general, and the usual comparison is
rigged.** Counting **coefficients**, the DFT appears to hold 0.765 of a Gaussian bump's energy in
8 coefficients against the DCT's 1.000000, which reads as decisive. But a DFT coefficient of real
data is complex and a DCT coefficient is not, so counting bins hands the DFT twice the storage.
Counted in **real numbers**, the same measurement is 0.999999 against 1.000000: a tie to six
digits. Recounted properly, the DCT's advantage turns out to be **entirely a boundary effect**:
decisive on a ramp (99.99 percent against 95.6 at 8 numbers), a tie on a centred bump, absent on
random data, and on an exactly periodic signal the DCT is the **worse** transform, discarding
$2\times10^{-3}$ where the DFT discards nothing.

**Modified Gram-Schmidt on functions does not lose orthogonality, and the textbook claim is about
the classical form.** The first draft repeated the standard statement that Gram-Schmidt degrades.
Measured on the Legendre weight, the **modified** form stays at $2\times10^{-16}$ out to degree 28
while the **classical** form reaches $7.8\times10^{-7}$, a factor of $3.8\times10^9$. The
implementation is the modified form, so the accuracy objection did not apply to it, and the
docstring and module header were corrected to say so. **The cost objection stands for both**: 435
integrals against Stieltjes's 56 at degree 28.

**The Wiener filter loses to a well chosen brick wall filter on a band limited signal.** With the
**true** signal power spectrum it wins every case, reaching 0.0085 against a brick wall's 0.0237.
With the spectrum estimated from the noisy data, which is what can actually be computed, it costs
a factor of 2 to 5 and scores 0.042 against 0.024. It wins by a factor of 3 on a square wave,
where the brick wall rings so badly that it is **worse than doing nothing** (0.202 against the
noisy signal's own 0.095). All four numbers are now reported rather than the convenient two.

**Zig-zag ordering saves nothing in the number of run length pairs, and 6 to 15 percent in bits.**
The obvious measurement, counting (run, value) pairs, gives a ratio of exactly 1.000 on every
image and quality tried, because the pair count is the nonzero count plus one and reordering does
not change it. The saving is in the **entropy coded bits**, and on one image class zig-zag is
**worse**: axis aligned edges put their DCT energy along the first row and column, which raster
order groups and zig-zag scatters, costing 4 to 5 percent.

**The near-minimax factor stays below 3 on every non-degenerate case, and the apparent
counterexamples are all artefacts.** Run without filtering, `cosh` reports 5.000 at degree 16 and
$t^2$ reports 5.800 at degree 16, which look like a decisive answer to the exercise. Both are
degenerate: an even function's best degree $2k$ and degree $2k+1$ approximations coincide, so the
truncation is dropping a coefficient that is exactly zero and the comparison is between different
degrees. Filtering those out and the roundoff-floor rows, the largest genuine factor found across
eleven functions and seven degrees is **3.305**, on $|t-0.3|$ at degree 16.

**The Pade advantage does not fall as a singularity approaches; that measurement was of the wrong
thing.** With the interval fixed at $[-0.5, 0.5]$ and the branch point of $\log(1-t/a)$ moved from
$a = 8$ to $a = 0.55$, the gain over Taylor falls from 3527 to 15, which reads as the method
failing near a singularity. With the interval **scaled** to the singularity, so the relative
distance is constant, the gain is **520.2 in every row to four digits**. The first table was
measuring the difficulty of the problem; the advantage depends only on the ratio of the interval
width to the distance, and on nothing else.

**A fitted decay order must skip the coefficients that are exactly zero.** An even function has
every odd coefficient at the roundoff floor, and including them mixes a decaying sequence with a
flat one. The DCT's decay order on a ramp then reads 2.04 at $n = 64$, 1.44 at $n = 128$ and 0.99
at $n = 256$, which looks like the theory failing as $n$ grows. Restricted to the coefficients
that carry anything, it is 2.03 at every size. The same correction was needed in
`chebapprox.coefficient_decay`, where the Runge function's rate read 0.985 instead of its true
1.22, and where `|t|` was classified as geometric when it is not.

**The Chebyshev coefficient decay has three cases, not two.** The first draft distinguished
geometric from algebraic. An **entire** function such as $\exp$ is neither: its coefficients fall
like $1/(2^kk!)$, faster than any fixed rate, and the local ratio $|a_k/a_{k+1}|$ keeps growing.
`coefficient_decay` now detects that and reports it separately, and it also reports how many
coefficients the fit used, because the classification needs the asymptotic regime to have started:
$\sqrt{t^2+0.01}$ is misclassified as algebraic at degree 20 and 40 and correctly as geometric
from degree 80.

**Five defects, found and fixed.**

- **`nalib.fft.fft_radix2` was silently wrong at every size.** A numpy slice is a view, so writing
  the first half of the butterfly result overwrote the values the second half still needed. The
  transform was off by an $O(1)$ amount, the butterfly count was still exactly right, and only a
  comparison against `nalib.dft.dft` caught it.
- **`nalib.chebapprox.to_power_basis` started the Chebyshev recurrence at $k = 0$**, giving
  $T_1 = 2x$ instead of $x$ and a factor of 2 in every later member. The round trip against
  Clenshaw disagreed by 67 percent at every degree.
- **`nalib.approx.remez` stopped after one iteration on an odd function over a symmetric
  interval.** The symmetric starting reference makes the first solve return $E = 4.9\times10^{-17}$
  and its error then has $n+1$ alternating extrema rather than $n+2$, so the exchange has nothing
  to exchange. On $\sin(3t)$ at degree 7 it returned $3.37\times10^{-4}$ against the correct
  $1.69\times10^{-4}$. Shifting the starting reference off centre fixes it.
- **`nalib.approx.remez` cycled for 60 iterations on a degenerate degree.** On $\cosh$ at degree 4,
  where there are $n+3$ extrema, dropping the globally weakest one breaks the alternation. Dropping
  from an **end** preserves it and converges in 4.
- **`nalib.orthopoly`'s Chebyshev orthogonality measured 0.5 instead of $10^{-16}$**, because a
  uniform grid in $x$ cannot integrate the singular weight $1/\sqrt{1-x^2}$. Each family now gets
  the substitution that removes its own difficulty, and Laguerre's needed a third attempt: the
  obvious $x = -\log u$ substitution puts almost no points where $x$ is large, and the integrand
  grows like $x^{2k}$ there, giving $5\times10^{-2}$ instead of $10^{-16}$.

**One correction to an earlier part.** Lesson 51's key takeaway claimed the natural spline is
$O(h^6)$ in the middle, from a fit over $n = 4$ to 64. Part 7's solution 51.4.1 refined that to
$n = 1024$ and found it settling at 4.01, 3.99, 3.91: the apparent 5 to 9 is the end condition's
$O(1)$ moment error still decaying geometrically in knot index. The lesson now carries the
corrected statement, so the lesson and its solution agree.

## D. What is deliberately not claimed

- **The measured near-minimax factor is not a bound.** It is between 1.47 and 3.31 over the
  functions and degrees tried, and the theory says it grows like $\log n$ without limit. "About 2"
  is a statement about the degrees people use, not a theorem, and the solutions say so.

- **The split radix comparison is stated in two units and they disagree.** Counting complex
  multiplications by the recursion gives a saving over radix-2 of 1.73 at $n = 2^{20}$, still
  falling. The published figure, in real flops, is $5/4 = 1.25$ asymptotically. The two differ
  because counting multiplications alone ignores the additions, which is most of the work, and the
  solutions report both rather than picking the flattering one.

- **The Gibbs constant is measured as 0.0893 against the exact 0.0894899**, and the discrepancy
  does not shrink with refinement. The overshoot peak narrows at the same rate as the grid
  refines, so the discrete signal never samples the continuous maximum. Reporting the limit as
  0.08949 from this data would be wrong, and calling the $2\times10^{-4}$ residual a failure would
  also be wrong.

- **Robust Pade helps in one row of fifteen and ties in the rest.** At $[14/14]$ with $10^{-11}$
  coefficient noise it is 381 times better, because the plain solve happens to put a real pole
  inside the evaluation interval. Everywhere else it neither helps nor hurts, and with no noise at
  $[10/10]$ it is very slightly worse. It is insurance, not an improvement, and the solutions say
  which.

- **Froissart doublets did not appear on `exp` with relative coefficient noise at any degree or
  precision tried.** They appear immediately with **absolute** noise, because `exp`'s coefficients
  decay factorially and relative noise stays ordered while absolute noise dominates the tail. The
  demonstration uses absolute noise and says why.

- **The rate distortion curve is measured on a smooth synthetic image**, which compresses far
  better than a photograph: 1.06 bits per pixel at quality 1, where a photograph would be nearer
  0.15. The **shape** of the curve, and the knee between quality 10 and 30, are representative; the
  absolute numbers are not, and the solution says so.

- **The SVD comparison in solution 60.5.3 quotes a figure that is not measured here.** The claim
  that a block DCT pipeline reaches roughly 35 dB where a truncated SVD reaches 22 to 25 dB at
  equal storage on a photograph is cited from the literature, not run in this repository, because
  no photograph is committed to it. The four **reasons** given for the gap are argued rather than
  measured.

- **Nothing here is timed.** Every cost claim is an operation count, either derived or counted by
  instrumenting the code. The FFT's 105000-fold advantage at $n = 2^{20}$ is a ratio of
  multiplication counts, and on real hardware memory traffic dominates arithmetic at that size, so
  the wall clock ratio would be smaller and machine dependent.

- **The DCT-via-FFT agreement degrades to $7.8\times10^{-14}$ at $n = 1024$**, and the degradation
  is in the **reference** rather than in the fast transform. The direct DCT sums $n$ terms per
  coefficient so its own error grows like $\sqrt n\varepsilon$. No claim is made that the fast
  version is losing accuracy; the opposite is measured for the FFT in solution 59.4.1.
