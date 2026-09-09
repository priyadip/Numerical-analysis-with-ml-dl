# Audit: Part 10, Ordinary Differential Equations

A part-level audit. [FINAL_AUDIT.md](FINAL_AUDIT.md) carries the current overall position and
links to every part-level audit, so that list is maintained in one place rather than repeated
here and going stale each time a part lands.

Everything below is measured from the repository by [`run_all.py`](run_all.py). No number in
this document was typed in by hand.

---

## Verdict

**Part 10 passes.** All ten lessons exist, all are executed, tested, numerically verified and
fully solved, with no placeholders, no unpaired files and no hardcoded dimensions.

| Check | Result |
|---|---|
| Lessons written | **10 of 10** |
| Notebooks executed end to end | **76 of 76 pass** (every notebook in the repository) |
| Automated tests | **8436 of 8436 pass** (782 new) |
| Assertions inside the Part 10 lessons | **37, all passing** |
| Concepts covered from the sources | **47 new, 431 of 507 total** |
| Concepts outstanding in Part 10 | **0** |
| Worked exercise solutions | **210**, one for every exercise |
| Generality scan | **0 candidates across the repository** |
| Mangled escape scan | **0 control characters across the repository** |

---

## A. Curriculum completeness

Part 10 was planned as 10 lessons. **All ten exist.**

| Lesson | Title | Figures | Assertions | Words | Exercises |
|---|---|---:|---:|---:|---:|
| 67 | Initial Value Problems and Euler's Method | 1 | 5 | 2543 | 21 |
| 68 | Taylor and Picard Methods | 1 | 2 | 2055 | 21 |
| 69 | Runge-Kutta Methods | 1 | 5 | 2018 | 21 |
| 70 | Adaptive Step Size Control | 1 | 3 | 2204 | 21 |
| 71 | Multistep Methods | 1 | 2 | 2014 | 21 |
| 72 | Stability and Stiff Equations | 1 | 5 | 2139 | 21 |
| 73 | Systems and Higher Order Equations | 1 | 3 | 1964 | 21 |
| 74 | Geometric and Symplectic Integrators | 1 | 3 | 1780 | 21 |
| 75 | Boundary Value Problems | 1 | 4 | 2319 | 21 |
| 76 | Collocation and Finite Elements | 1 | 5 | 2255 | 21 |
| **total** | | **10** | **37** | **21291** | **210** |

**Nothing planned was dropped.**

Ten new library modules back the part:

| Module | Lines | Public functions | Test lines | Tests |
|---|---:|---:|---:|---:|
| `ivp` | 424 | 12 | 276 | 77 |
| `taylorode` | 268 | 7 | 209 | 35 |
| `rungekutta` | 499 | 13 | 334 | 128 |
| `adaptivestep` | 420 | 13 | 236 | 58 |
| `multistep` | 599 | 18 | 287 | 133 |
| `stability` | 528 | 16 | 330 | 79 |
| `odesystems` | 443 | 16 | 265 | 60 |
| `symplectic` | 463 | 17 | 235 | 47 |
| `bvp` | 1131 | 24 | 491 | 82 |
| `femode` | 819 | 24 | 368 | 83 |
| **total** | **5594** | **160** | **3031** | **782** |

The solutions are [`part10_ordinary_differential_equations.md`](../solutions/part10_ordinary_differential_equations.md):
8614 lines, 210 solutions, 200 executable code blocks, all of which run.

## B. Source completeness

Part 10 covers **47 of 47** concepts attributed to it, so the part is
marked complete in [`coverage_report.md`](coverage_report.md).

| Source | Chapter | Covered in Part 10 | Covered repository wide |
|---|---|---|---|
| Sauer | 6 Ordinary Differential Equations | 19 of 19 | 19 of 19 |
| Sauer | 7 Boundary Value Problems | 7 of 7 | 7 of 7 |
| Gupta | 14 First Order ODE IVPs | 13 of 13 | 13 of 13 |
| Gupta | 15 ODE Systems and BVPs | 4 of 4 | 6 of 6 |
| Supplementary | four topics no source covers | 4 of 4 | 4 of 4 |

Sauer chapters 6 and 7 and Gupta chapter 14 are now **complete across the whole repository**.
Gupta chapter 15 is complete too, with four of its six concepts here and the other two in lesson
61, where the finite difference formulas they name belong.

The four supplementary concepts are:

- Butcher tableau notation and Runge-Kutta order conditions, lesson 69
- Root condition, zero-stability and the Dahlquist barriers, lesson 71
- Regions of absolute stability, A-stability and L-stability, lesson 72
- Symplectic Euler and Stormer-Verlet, energy behaviour over long times, lesson 74

None of them appears in a supplied source, and a graduate treatment of this material without them
would be incomplete.

## C. What the measurements changed

Every claim in the ten lessons was checked against a measurement, and where the two
disagreed the question asked was which of them was wrong. Below are the 35 cases
where the answer turned out to be the code or the claim rather than the test, in 4
groups: answers that looked right, claims that measured the wrong quantity, docstrings that
had drifted from the code they describe, and exercises whose expected answer did not survive
being run.

### Wrong answers that looked right

**`lipschitz_near` reported the $y^{2/3}$ singularity as bounded.** It compared the estimate at
successive radii and asked whether the ratio grew. Over five decades of radius the constant rises
only forty six fold, which reads as settling. Replaced with a fitted growth exponent, which comes
out at exactly $-1/3$ for $y^{2/3}$ and $-1/2$ for $\sqrt{\lvert y\rvert}$: **the exponent
identifies which singularity it is, and the ratio cannot.**

**`rungekutta.order_conditions` checked the order conditions in floating point.** RK4's first
condition sums to $0.9999999999999999$, so the only available verdict was "within a tolerance I
chose". Rewritten with an exact rational path: given the tableau as `Fraction` objects, every
condition's miss is an exact zero or an exact nonzero. The nine order 5 conditions were added at
the same time, and RK4 misses them by $-1/80$ and $\pm 1/120$ and $\pm 1/240$, which is not a
rounding artefact. Dormand-Prince and Fehlberg satisfy all seventeen to $10^{-16}$, which is the
check that the new conditions are the right ones.

**`odesystems.reduction_is_faithful` checked the reduction by differencing.** It compared
`np.gradient` of the computed first component against the second and reported $8\times10^{-4}$
falling like $h^2$, which looks like a second order property of the reduction. It is a measurement
of `np.gradient`. The reduction guarantees the **exact** identity $f(t,u)_1 = u_2$ at every state;
checked that way the residual is $0.0$ at every step count.

**`uniqueness_fails_on` had the same fault**, reporting $4\times10^{-6}$ for four functions that
satisfy $y' = y^{2/3}$ identically. Checked against the closed form derivative the residual is
$6\times10^{-17}$.

**`symplectic`'s step change demonstration claimed the energy bands were disjoint.** The measured
bands were **nested**, every one of them with its high end at exactly $0.0$, because the initial
condition sits at the top of all of them. The claim was false and the conclusion drawn from it was
too. Replaced with a measurement of what actually breaks: a fixed step and a deterministic
alternation both stay bounded, and only an **irregular** step drifts. That is a different and
sharper statement, and it is the one that explains why adaptivity is the obstruction.

**`bvp.linear_shooting` was documented as dividing by zero at a resonance.** It does not. The
homogeneous solution vanishes at $b$ in exact arithmetic and what RK4 returns instead is its own
$O(h^4)$ truncation error, so the correction is divided by a number that **shrinks as the step is
refined**. The returned answer grows from $8\times10^{6}$ to $5\times10^{11}$ across a sweep, and
nothing complains. The function now reports `near_a_resonance`, and
`at_a_resonance_refining_makes_it_worse` measures the whole effect, fitting the end value's decay
at order 3.999.

**`bvp`'s wiggle detector counted sign changes with no threshold**, reporting 20 wiggles in a
solution that is monotone to fifteen digits, because the flat part's successive differences have
arbitrary signs at the $10^{-16}$ level. Replaced with a violation of the discrete maximum
principle, which is the property that actually fails, and it now agrees with the M-matrix condition
exactly at every grid size.

**`femode.hat` returned zero at its own node**, for node 0 only. The right branch was half open, so
the first hat function had no left half to set its own value. The basis was not a partition of
unity at exactly one point of every grid, and the assembled matrix was wrong in its first row.

### Claims that were the wrong quantity

**`multistep`'s error constant ratios were quoted as 2.4, 4.6 and 7.3.** Measured, they are 5, 9,
13.2 and 17.6. The step count was also computed from `alpha` alone, which is wrong for
Adams-Moulton where `beta` is longer, and the unstable example's parasitic root is $-5$ rather than
$-2$ with order 3 rather than 2.

**`stability.classify` tested L-stability against a fixed threshold of $10^{-6}$.** BDF2's growth
factor decays like $\lvert z\rvert^{-1/2}$, so at $z = -10^{10}$ it is still $7\times10^{-6}$ and
the threshold called an L-stable method not L-stable. Replaced with "small **and still falling**"
across two magnitudes, which is what "tends to zero" means and needs no number chosen in advance.

**`second_dahlquist_barrier` returned `None` for BDF3**, because `classify` only knows one step
methods and BDF3 has three. The barrier could not be checked on the methods it is about. Added
`multistep_roots`, `multistep_is_stable_at` and `stability_angle`, which work from
$\rho(w) - z\sigma(w)$ for a method of any step count, so zero-stability is now literally the
$z = 0$ case of absolute stability. The barrier is now measured rather than asserted: the highest
A-stable order among eight named methods is 2.

**`explicit_stability_is_not_the_answer` reported one number called `spread`**, which was the ratio
of stability limits (1.39) while the surrounding text read it as the ratio of evaluations (2.88).
Both are now reported, along with which method is cheapest.

**`bvp.upwinding_removes_the_wiggle` fitted the order over the whole sweep**, giving 1.82 for a
second order method and 0.55 for a first order one. The head of the sweep is a different regime,
not a slow start: the grid is not resolving the layer at all there. Fitting only the rows with a
cell number below 0.1 gives 2.000 and 0.966, and the function now reports both so the difference is
visible.

**`bvp.supraconvergence_*` measured the truncation error in the max norm**, reading 0.88 on an
alternating grid and 0.62 on a random one for a quantity whose order is 1. The max norm picks
whichever node has the largest third derivative and mixes the second order part of the error in
with the first. The rms norm reads 0.99 on both.

**`bvp.existence_can_fail` measured the distance from $\pi^2$.** The discrete problem resonates at
$(4/h^2)\sin^2(\pi h/2)$, which is below $\pi^2$ by $\pi^4h^2/12$, and measuring from the wrong one
made the $1/\text{gap}$ law look broken: the product $\text{cond}\times\text{gap}$ spreads by 3.03
using $\pi^2$ and by exactly **1.000** using the discrete value.

**`bvp.graded_grid_for_a_layer` graded towards the wrong end.** For $\varepsilon y'' + y' = 0$ the
layer is at the inflow boundary, and clustering at the outflow one makes the error 2.6 times larger
than not clustering at all. The direction is now an explicit argument.

### Where the code was right and the docstring was not

**`blows_up_at` said refining moves the reported blow up later.** It moves it **earlier**, from
1.2800 at 50 steps to 1.0225 at 800, halving the overshoot each time the step halves and converging
on $t = 1$ from above at Euler's own first order.

**`euler_step_size_floor` claimed a floor near $\sqrt{\varepsilon}$.** At $h = 5.96\times10^{-9}$,
a quarter of $\sqrt\varepsilon$, the error ratio is still exactly 4.000 per refinement. Rounding
errors accumulate like $\sqrt n$ rather than $n$, so the crossing is at $\varepsilon^{2/3}$ and no
realistic run reaches it. Confirmed from the other side in solution 67.3.5, where single precision
puts the crossing at $2.4\times10^{-5}$ and the measured minimum lands within a factor of 3 of it.

**`start_with` said the run inherits the starter's order.** It achieves $\min(p, q+1)$: AB4 started
by Euler runs at 1.94, by Heun at 2.94, by Kutta's third order rule at 3.89.

**Adaptive stepping detected first-same-as-last and did not use it.** With the stage reuse implemented, Dormand-Prince costs 6.39 evaluations per accepted step. Turning off local extrapolation puts it back to 7.39, because the seventh stage is evaluated with the high order answer and returning the low order one makes that stage useless. That is the second and less obvious cost of the honest solver, and it was invisible before the reuse existed.

**`tacoma_narrows` was documented with a critical wind speed.** The measured growth is not monotone:
0.62, 0.59, 0.73, 1.31, 17.6, 4.39 across a sweep from 40 to 200. It first exceeds 1 near 100,
peaks near 140 and falls again by 200, so what the sweep passes through is a **resonance** and not
a threshold.

### Solutions where the exercise's expectation did not survive

**Exercise 69.4.1 asks whether RK4 still wins on a problem with a discontinuous fourth
derivative.** The obvious test problem is degenerate five separate ways: the right hand side depends
only on $t$, so every method collapses to a quadrature rule; Kutta's third order rule becomes
Simpson's rule; $\lvert t-1\rvert^3$ is a cubic on each side; the kink lands on a grid point;
and Euler and Heun coincide because $g(0) = g(2)$. The resulting table of exact zeros looks like a
result. Only a manufactured problem with genuine $y$ dependence separates the methods.

**Exercise 70.3.5 asks to compare two step controllers.** Before the comparison is possible there
is a larger finding: starting from $h_0 = 0.05$ with a growth cap of 5, the controller leaps the
pulse in one stride and **returns an answer 3.957 away from the truth while reporting success**, at
every tolerance down to $10^{-10}$.

**Exercise 70.4.2 expects local extrapolation to make the tolerance less meaningful.** It measures
the reverse. With extrapolation the ratio of achieved error to tolerance is roughly constant, 1.6
to 3.7 over eight decades; without it the ratio **grows** from 4.3 to 252, at 1.62 to 1.74 per
decade, which is the $\text{tol}^{-1/5}$ the orders predict.

**Exercise 70.5.2 asks for an example of order reduction with an explicit method.** There is none,
and the reason is the answer: seeing it needs $\lambda h$ large across the whole sweep, which for an
explicit method is outside the stability region, so the runs diverge rather than losing order. The
effect appears instead in solution 72.3.5, on a Rosenbrock method, which is implicit enough to run
there.

**Exercise 71.5.2 asks how a predictor-corrector pairing suppresses Milne-Simpson's parasitic
mode.** With an AB2 predictor it does not: the pair's parasitic root is **further** outside the
circle than the corrector's, 1.0067 against 1.0033 at $\lambda h = -0.01$ and 1.312 against 1.178 at
$-0.5$.

**Exercise 73.5.3 asks for a system where a non quadratic invariant is better conserved than a
quadratic one.** The construction attempted does not separate them: on a two dimensional oscillator
both drift by exactly the same relative amount, because RK4 damps both amplitudes by the same factor
and every smooth function of them inherits it. The forward direction is demonstrated instead, with
the two stage Gauss method conserving angular momentum to $10^{-14}$ while its energy drifts at
$10^{-11}$, a separation of 4600 against RK4's 5.

**Exercise 73.3.5 asks whether a ramped wind takes off where the fixed sweep says.** It takes off
**later**, or not at all within 300 time units, because growth takes time and a ramp passes through
the resonance too quickly to accumulate it.

**Exercise 73.4.1 expects the closure gap to scale like $\left(\frac{1+e}{1-e}\right)^4$.** The
fitted exponent is **6.96**, because the eccentricity enters twice: once through the local error at
perihelion and once through the energy error changing the period and turning a position error into
a phase error.

**Exercise 74.3.4 expects the projected method to lose on the trajectory and Verlet to win.**
Verlet's phase error is the **largest** of the three at the same step, because it is second order
against RK4's fourth. What it has is the absence of a trend, which at that run length has not yet
paid for the order.

**Exercise 74.5.3 expects the projected map to be non symplectic.** Its Jacobian determinant is not
merely different from 1; it is essentially **zero**, because the projection sends a two dimensional
neighbourhood onto a one dimensional level set.

**Exercise 75.5.2 asks for the Shishkin mesh's $\varepsilon$ independent bound.** With central
differences it is not $\varepsilon$ independent at all: the constant runs 0.97 at
$\varepsilon = 10^{-2}$ and 1875 at $10^{-4}$, because the coarse half's cell number reaches 312.
With **upwinding** on the same mesh the constant is 0.63, 0.69, 0.72 across every combination
tried, and the bound is first order.

**Exercise 76.3.4 expects cubic B-spline collocation to be fourth order.** Collocated at evenly
spaced sites, with exact derivatives from the standard recursion, it is **second** order. The
fourth order version needs the two Gauss points of each element: the basis and the collocation
points are two separate choices.

**Exercise 76.4.1 asks to measure the constant in Cea's lemma.** On $-u'' = f$ the ratio is exactly
1 for a reason that has nothing to do with the lemma: the Galerkin solution **is** the interpolant,
by the nodal exactness of exercise 2.4. A reaction term is needed before the two functions differ
at all.

## D. Negative results, reported as such

Four exercises in this part end without the answer they ask for, and each is marked in the
solutions rather than replaced with a problem that would have worked.

| Exercise | What was asked | What was measured |
|---|---|---|
| 69.4.2 | where RK4 overtakes Heun | RK4 is cheaper at every tolerance tried; the crossover is below the asymptotic regime and means nothing |
| 70.5.2 | order reduction with an explicit method | cannot be exhibited: the runs that would show it diverge |
| 71.5.2 | how the pairing suppresses the parasitic mode | this pairing makes it worse |
| 73.5.3 | a system where the reverse holds | the construction does not separate the two invariants |

## E. What the exact arithmetic bought

Two places in this part were rewritten to work in exact rational arithmetic, and both changed the
kind of statement available rather than only its precision.

**Runge-Kutta order conditions.** In floating point RK4's first condition sums to
$0.9999999999999999$ and the worst residual across the classical tableaux ranges from $0.5$ for
Euler to $1.1\times10^{-16}$ for RK4. A reader given only those numbers has to decide where between
them to draw a line. In exact arithmetic each condition's miss is $0$ or a specific rational, and
there is no line to draw. `all_tableaux_are_what_they_claim` now reports both verdicts, and the
point of running both is that the float column cannot certify itself.

**Adams and BDF coefficients.** Both families are generated as `Fraction` lists, so AB4's weights
are exactly $\tfrac{1}{24}(55, -59, 37, -9)$ and BDF3's are exactly
$(1, -\tfrac{18}{11}, \tfrac{9}{11}, -\tfrac{2}{11})$ with $\beta_0 = \tfrac{6}{11}$. Solution
71.2.1 derives AB3 by integrating the Lagrange basis symbolically and asserts equality with the
library's rationals, which is an equality check rather than a tolerance check.

## F. The two Dahlquist barriers, measured

Both barriers are checked against the repository rather than quoted.

**The first barrier** is searched: a zero-stable $k$ step method has order at most $k+2$ for even
$k$ and $k+1$ for odd $k$. The measured maxima are 2, 4, 4, 6, 6, 8, 8, 10 for $k = 1$ to 8, so
every second extra step buys nothing.

**The second barrier** is measured from each method's own coefficients, through the roots of
$\rho(w) - z\sigma(w)$, so a three step method is tested the same way a one step method is. Across
eight named methods the highest A-stable order found is **2**.

**What the BDF family settles for instead** is measured by bisection on the stability angle,
against the published table:

| order | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| measured angle | 90.000 | 90.000 | 86.112 | 73.392 | 51.880 | 18.006 | 0.000 |
| published | 90.00 | 90.00 | 86.03 | 73.35 | 51.84 | 17.84 | 0.00 |

The worst gap is 0.17 degrees. BDF7 has no angle at all: it is not stable even on the negative real
axis, which is the same zero-stability failure as lesson 71's counterexample.

## G. Trefethen and Bau

**Nothing in Part 10 is attributed to Trefethen and Bau.** The book covers numerical linear algebra
and has no material on differential equations, and the concept map assigns none of Part 10's
concepts to it. Where Part 10 uses linear algebra it cites the earlier parts that took it from
Trefethen and Bau: lesson 72's eigenvalue view of stiffness points at Part 6, and lesson 76's
symmetric positive definite assembled matrix points at Part 3's Cholesky and Part 5's conjugate
gradients.

## H. What is not here

Three things a full treatment would include and this part does not.

**Implicit Runge-Kutta methods are exercises rather than library code.** The two stage Gauss method
and Radau IIA are built and measured in solutions 69.3.3, 72.3.4 and 73.5.3, and neither is in
`nalib`. The reason is scope: an implicit stage solver needs a Newton loop with a Jacobian per
stage, which is Part 5's subject applied inside Part 10's, and the exercises reach the results
without the machinery.

**Differential algebraic equations are absent entirely.** They are not in the source chapters and
they need index reduction, which is a subject of its own.

**Adaptive multistep methods are exercise 71.3.3 and 71.3.4 rather than library code.** Milne's
device and variable step coefficient generation are both built and measured there, and the honest
reason they stop at exercises is exercise 71.5.3's finding: the whole stability theory of the lesson
assumes a fixed step, and a variable step method needs a different one.

## I. Reproducibility

Every lesson fixes `rng = np.random.default_rng(42)` in the standard preamble, and every function in
the ten new modules that uses randomness takes a `seed` argument defaulting to 42. Re-running
`verification/run_all.py` reproduces every number in this document.
