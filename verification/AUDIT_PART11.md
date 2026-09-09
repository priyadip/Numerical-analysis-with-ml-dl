# Audit: Part 11, Partial Differential Equations

A part-level audit. [FINAL_AUDIT.md](FINAL_AUDIT.md) carries the current overall position and
links to every part-level audit, so that list is maintained in one place rather than repeated
here and going stale each time a part lands.

Everything below is measured from the repository by [`run_all.py`](run_all.py). No number in
this document was typed in by hand.

---

## Verdict

**Part 11 passes.** All seven lessons exist, all are executed, tested, numerically verified and
fully solved, with no placeholders, no unpaired files and no hardcoded dimensions.

| Check | Result |
|---|---|
| Lessons written | **7 of 7** |
| Notebooks executed end to end | **84 of 84 pass** (every notebook in the repository) |
| Automated tests | **8933 of 8933 pass** (448 new) |
| Assertions inside the Part 11 lessons | **110, all passing** |
| Concepts covered from the sources | **32 new, 466 of 507 total** |
| Concepts outstanding in Part 11 | **0** |
| Worked exercise solutions | **147**, one for every exercise |
| Generality scan | **0 candidates across the repository** |
| Mangled escape scan | **0 control characters across the repository** |

---

## A. Curriculum completeness

Part 11 was planned as 7 lessons. **All seven exist.**

| Lesson | Title | Figures | Assertions | Words | Exercises |
|---|---|---:|---:|---:|---:|
| 77 | PDE Classification and Stencils | 1 | 21 | 2897 | 21 |
| 78 | Parabolic Equations | 1 | 22 | 2689 | 21 |
| 79 | Consistency, Convergence and Stability | 1 | 15 | 2324 | 21 |
| 80 | Multidimensional Parabolic Problems and ADI | 1 | 11 | 1987 | 21 |
| 81 | Hyperbolic Equations and the CFL Condition | 1 | 14 | 2068 | 21 |
| 82 | Elliptic Equations | 1 | 13 | 1984 | 21 |
| 83 | Nonlinear PDEs and the Method of Lines | 1 | 14 | 2276 | 21 |
| **total** | | **7** | **110** | **16225** | **147** |

**Nothing planned was dropped.**

Seven new library modules back the part:

| Module | Lines | Public functions | Test lines | Tests |
|---|---:|---:|---:|---:|
| `pdeclass` | 798 | 25 | 379 | 75 |
| `parabolic` | 932 | 20 | 349 | 75 |
| `pdestability` | 657 | 14 | 249 | 60 |
| `adi` | 688 | 18 | 285 | 52 |
| `hyperbolic` | 677 | 18 | 305 | 77 |
| `elliptic` | 765 | 20 | 333 | 61 |
| `nonlinearpde` | 769 | 13 | 286 | 48 |
| **total** | **5286** | **128** | **2186** | **448** |

The solutions are [`part11_partial_differential_equations.md`](../solutions/part11_partial_differential_equations.md):
6965 lines, 147 solutions, 121 executable code blocks, all of which run.

## B. Source completeness

Part 11 covers **32 of 32** concepts attributed to it, so the part is
marked complete in [`coverage_report.md`](coverage_report.md).

| Source | Chapter | Covered in Part 11 | Covered repository wide |
|---|---|---|---|
| Sauer | 8 Partial Differential Equations | 11 of 11 | 11 of 11 |
| Gupta | 16 Partial Differential Equations | 19 of 19 | 19 of 19 |
| Supplementary | two topics no source covers | 2 of 2 | 2 of 2 |

Sauer chapter 8 and Gupta chapter 16 are both **complete across the whole repository**, which
closes the last partial differential equation material in either source.

The two supplementary concepts are:

- Lax equivalence theorem, lesson 79
- Method of lines linking PDEs to ODE solvers, lesson 83

Neither appears in a supplied source, and a graduate treatment of this material without them would
be incomplete: the Lax equivalence theorem is the reason consistency and stability are the two
things worth measuring, and the method of lines is what connects this part to Part 10.

## C. What the measurements changed

Every claim in the seven lessons was checked against a measurement, and where the two
disagreed the question asked was which of them was wrong. Below are the 33 cases
where the answer turned out to be the code or the claim rather than the test, in 4
groups: answers that looked right, claims that measured the wrong quantity, standard warnings
that could not be reproduced, and exercises whose expected answer did not survive being run.

### Wrong answers that looked right

**`pdeclass.stencil` indexed the wrong thing and every order it reported was wrong.** It called
`fornberg(...)` and took element `[k]` of the result, but `fornberg` returns the one dimensional
weight array, not a table of derivative orders. Every stencil built through it was therefore the
weights for the wrong derivative, and `stencil_order` dutifully measured the order of the wrong
formula. Fixed by returning the weights directly, and the Taylor moment tolerance was rescaled to
$\sum \lvert w_i\rvert\,\lvert s_i\rvert^{m}$ so it means the same thing at every stencil width.

**The nine point Laplacian's order was stated wrong twice, in opposite directions.** The usual
sentence is that it is fourth order. Measured, it is **second** order on a general function and
**sixth** on a harmonic one. One identity explains both: for harmonic $u$, $u_{xxyyyy} =
-u_{xxxxyy}$, so the $h^{4}$ term cancels as well as the $h^{2}$ one. The exactness test settles it
with no tolerance at all: the five point rule is exact on harmonic polynomials up to degree 3, the
nine point rule up to degree 7.

**`the_explicit_limit_is_sharp` was measuring nothing.** It started from a smooth initial profile,
whose amplitude at the worst mode $\phi = \pi$ is zero, so the mode the stability limit is about
was never excited and the run stayed bounded well past the limit. Rewritten with three sweeps, a
short smooth profile, a long smooth one, and a jump, and with a quantitative prediction
$\text{seed} \times \text{growth}^{\,\text{steps}}$ that is right in all 21 rows.

**`crank_nicolson_rings_on_a_step` counted wiggles without a threshold.** Any rounding level
oscillation counted, so the answer depended on the mesh rather than on the ringing. Replaced with
the signed amplitude of the worst discrete sine mode and a count of its sign flips, which is an
exact test with nothing to tune.

**`hyperbolic.travelling_bump` forced its boundary values to zero**, which is not what the exact
solution does there. That cost $3.1\times10^{-2}$ and hid the property the lesson is built on: at
$\lambda = 1$ the scheme is exact. With the correct boundary values the error at $\lambda = 1$ is
$6.4\times10^{-16}$.

**`changing_data_it_cannot_see` placed the bump where it could not matter for a second reason.**
Its contribution at the sample point was $1.8\times10^{-60}$, so the measured change was zero
because the data was negligible, not because the domain of dependence excluded it. Moved onto the
characteristic, the true change is $0.5$ and the computed change is **exactly** $0.0$, which is the
statement the exercise is about.

**`elliptic`'s test problem made conjugate gradients converge in one step.** The solution
$\sin \pi x \sin \pi y$ is an exact eigenvector of the five point matrix, so the whole Krylov space
is one dimensional at every grid size, and any scaling law measured on it is a measurement of that
degeneracy. Added `mixed_modes_problem`, $u = e^{x}\sin\pi x\,\sin\pi y$, which keeps the zero
boundary values and spreads the right hand side across the spectrum. Only then does the count grow
as $O(h^{-1})$, and a dedicated measurement of the degeneracy was added so the trap is documented
rather than quietly removed.

**`bratu_critical_value` was four times too large.** It computed $2(c/\cosh(c/4))^{2}$ where the
elimination gives $c^{2}/(2\cosh^{2}(c/4))$. Corrected to $3.513830719125161$, which solution 83.3.1
then reproduces by continuation, converging at second order in $h$.

**The pseudospectral radius was computed wrongly twice before it was right.** A Cartesian scan of
the complex plane missed the thin shell where the resolvent norm crosses the threshold and reported
the spectral radius for a normal matrix. A per-ray bisection from the origin found the wrong
connected component. Bisecting on the **radius**, asking whether any point of the circle of radius
$R$ is inside, gives exactly $\rho + \varepsilon$ on a normal matrix, which is the check that the
routine is right.

**The ADI refinement sweeps did not compare like with like.** They used `steps * k` as the end time,
so the coarse grids reached $t = 0.31$ and the fine ones $t = 0.048$, and the fitted order was a
measurement of that. Added `steps_for`, which rounds the step count up so every grid reaches the
same $t_{\text{end}}$.

**Solution 78.3.3 fitted order $0.018$ for a moving boundary problem.** The assembly used a scalar
where a diagonal was needed and had the sub and super diagonal convection signs swapped. Fixed, the
order is $1.9999$.

**The Newton-Krylov solver stalled at 321 points, and the tolerance was the reason.** A matrix free
Jacobian cannot drive the residual below the noise in its own difference quotient, and that floor
grows like $h^{-2}$ because the residual contains $1/h^{2}$ differences. Measured, the achievable
final residuals are $1.2\times10^{-12}$, $4.6\times10^{-12}$, $1.6\times10^{-11}$ on successive
grids. A fixed absolute tolerance of $10^{-11}$ converges up to 161 points and then spins for every
allowed iteration. The stopping test in solution 83.3.4 is relative to the scale of $F$.

**The BDF startup ramp caps every high order member at second order.** Ramping from BDF1 makes
BDF2, BDF3 and BDF4 all measure $2.00$, which looks like the methods working. With exact starting
values they measure $1.98$, $2.97$ and $3.93$. A single first order step has an $O(k^{2})$ local
error and that propagates as an $O(k^{2})$ global error no matter how good the rest of the run is,
so BDF4 was paying four times the storage of BDF2 to deliver BDF2's accuracy.

**The splitting order measurement read $1.0$ for Strang before it read $2.0$.** The diffusion half
was taken with a forward Euler step, whose own first order error dominated and hid the splitting
error completely. Solving the linear half by its matrix exponential and the logistic half by its
closed form isolates the splitting error, and then Lie measures $0.9992$ and Strang $2.0002$.

### Claims that were the wrong quantity

**Allen-Cahn coarsening is not a power law, and a power law fits it.** Evolved from noise, the
interface count falls $514 \to 188 \to 94 \to 32 \to 10 \to 4$ over four decades of time, which fits
$t^{-0.42}$ well. Then it stops dead: from $t = 10$ to $t = 1000$ the count does not change. The
right measurement is the lifetime of one configuration against its separation, and that is
exponential: $\log T$ against $d/\varepsilon$ has increments $1.0603, 1.2355, 1.3575, 1.4002,
1.4112$, converging on $\sqrt 2 = 1.41421$, which is the decay rate of the $\tanh$ interface tail.
The exponential fit's largest residual is $0.17$ against the power law's $0.57$. The stall is then
predicted rather than surprising: four interfaces at $d/\varepsilon = 12.5$ have an expected
lifetime near $e^{\sqrt 2 \times 12.5} \approx 4\times10^{7}$.

**Newton's convergence order does not depend on the strength of the nonlinearity.** Fitting the
whole residual history says it does, giving $1.75$ on the lesson's run and $1.1$ at the strongest
nonlinearity tried. Fitting the tail alone gives $2.00$ at every strength across five orders of
magnitude of the coefficient. What changes is the number of slow steps before the quadratic phase,
$1, 1, 2, 2, 2, 5, 7, 12, 18$, and the height of the residual spike, from $1$ to
$4.9\times10^{8}$.

**The extra points in the nine point Laplacian are not what buys fourth order.** Solved rather than
applied, the uncorrected nine point rule measures order $2.0021$ and is $2.2$ times **worse** in
absolute error than the five point rule at the finest grid. It is the Mehrstellen right hand side
correction, $f + (h^{2}/12)\Delta_h f$, that gives order $3.9927$ and a factor of $7000$ in accuracy
for one extra sweep.

**The cooling fin heat balance looked first order and is exact.** Summing $(u - u_\infty)h^{2}$ over
every node overcounts the boundary strip, giving a relative gap of $2.1\times10^{-1}$ falling like
$h^{1.02}$. With trapezoid weights the gap is $10^{-14}$ on eleven points a side and stays at
rounding: the balance is an identity that the ghost point treatment makes exact, not a property
that converges.

**The finite element method has no nodal superconvergence in two dimensions.** In one dimension the
P1 solution is exact at the nodes, measured here at $10^{-13}$ with no order at all. In two
dimensions the nodal error is second order and the ratio of the $L^{2}$ error over the elements to
the nodal maximum sits flat at about $1.2$ across four grids. The one dimensional proof uses the
fact that the Green's function is piecewise linear, which has no second dimension version.

**The exercise's own formula for the Fisher front speed is wrong.** Exercise 83.4.3 asks for the
measurement against $\min(Da + r/a,\ 2\sqrt{Dr})$. Since $Da + r/a \ge 2\sqrt{Dr}$ always, that
minimum is the constant $2\sqrt{Dr}$, and at $a = 1$ it predicts $0.2$ where the front travels at
$1.0100$: wrong by a factor of five. The correct rule is piecewise, $Da + r/a$ below $a^{*}$ and
$2\sqrt{Dr}$ above, and it matches the measurement to $0.00\%, 0.00\%, 0.01\%, 0.05\%$ in the
shallow regime.

**The explicit heat equation limit $r \le 1/2$ is conservative and the exact one is known.** The
largest discrete eigenvalue is $(4\alpha/h^{2})\cos^{2}(\pi h/2)$, not $4\alpha/h^{2}$, so the exact
limit is $1/(2\cos^{2}(\pi h/2))$. Measured, that is $0.512543, 0.503097, 0.500772, 0.500193$ on
successive grids: always above $1/2$, by $2.5\%$ at eleven points.

**$\kappa \approx 4/(\pi h)^{2}$ overestimates the condition number by a constant.** The exact value
is $\cot^{2}(\pi h/2) = 4/(\pi^{2}h^{2}) - 2/3 - O(h^{2})$, and the measured offset climbs
$0.6561, 0.6641, 0.6660, 0.6665$ toward $2/3$ while the residual after removing it shrinks by four
each row, which identifies the next term as well.

### Standard warnings that could not be reproduced

Four warnings that appear in the standard treatments were tested here and did not survive. Each is
reported as a negative result in the module that tests it, with the reason it fails rather than a
substitute example.

**Non-normality causes no transient growth in this scheme.** The convection-diffusion one step
matrix is not normal, and the standard warning is that its powers can grow before they decay.
Sweeping the **whole** stable region, the 2-norm comes out just below 1 every time, so the powers
decrease monotonically from the first step. What non-normality does cost is the delay before the
asymptotic rate takes hold, measured at 244 steps at $\text{Pe} = 1.9$.

**The ADI half step boundary warning could not be reproduced with a manufactured solution.** The
usual claim is that using $u(t + k/2)$ for the intermediate array drops the method from second order
to first. Eliminating the intermediate array shows the two candidate values differ by
$O(k^{2}h^{2})$ for anything that solves the equation, which is why a manufactured solution can
never show it. With boundary data prescribed independently and switching in time, the naive
treatment costs a **factor of 2** at every step count, not an order.

**Three dimensional splitting does not lose its order to the sweep sequence.** On a tensor product
grid the direction operators commute exactly, $\lVert AB - BA\rVert = 0$, so the splitting error has
no first order term and the orderings differ by $4\times10^{-17}$.

**There is no magic step for the two dimensional wave equation.** The one dimensional scheme is
exact at $\lambda = 1$, and the natural guess is $\lambda = 1/\sqrt 2$ in two dimensions. Measured,
that value is exact only for the $(1,1)$ mode and is a coincidence of it: general data is not
reproduced.

### Solutions where the exercise's expectation did not survive

**The harmonic polynomial test cannot separate the two nine point rules.** Both the compact and the
wide nine point Laplacian are exact on harmonic polynomials up to degree 7, though one is second
order and the other fourth. The test the exercise proposes is real and it does not measure what the
exercise wants it to.

**Isotropy is not an advantage here.** The nine point rule's error is direction independent, and
what that means in practice is that it is uniformly as bad as the five point rule's **worst**
direction rather than uniformly as good as its best.

**Red-black ordering does not change the Gauss-Seidel rate.** The two spectral radii are identical
to rounding at every size, and both equal $\rho_{\text{Jacobi}}^{2}$ exactly. What red-black buys is
that it is the consistent ordering the SOR theory needs, and that it is parallel, which is a wall
time gain and not a rate gain.

**Jacobi preconditioning cannot do anything to the five point matrix.** The diagonal is the constant
$4/h^{2}$, so $D^{-1}A$ is a scalar multiple of $A$ and the conjugate gradient iterates are
identical, not merely similar. The measured counts match exactly at every size.

**No preconditioner in the exercise is worth its cost on this matrix.** SSOR and incomplete Cholesky
make things worse at 15 unknowns a side and save $12\%$ and $23\%$ of the iterations at 63, against
roughly doubling the work per iteration.

**ADI's advantage over a Krylov solve is a constant, not an order.** The two dimensional
Crank-Nicolson matrix has $\kappa \approx 1 + 4r$, bounded independently of $h$, so conjugate
gradients needs $O(1)$ iterations on it.

## D. Negative results, reported as such

Nine results in this part are negative, and each is marked in the module or the solution rather
than replaced with a problem that would have worked.

| Where | What was asked | What was measured |
|---|---|---|
| `pdestability` | the cost of non-normality | no transient growth anywhere in the stable region; the cost is a 244 step delay |
| `adi` | the order loss from the half step boundary | a factor of 2, not an order, and only with data that does not solve the equation |
| `adi` | the splitting error from the sweep order in 3D | exactly zero: the operators commute on a tensor grid |
| `hyperbolic` | the two dimensional magic step | none exists; $\lambda = 1/\sqrt2$ is exact for one mode only |
| `hyperbolic` | negative group velocity | never negative for this scheme, unlike leapfrog advection |
| `elliptic` | the gain from red-black ordering | none in the rate; the radii are identical |
| `elliptic` | which preconditioner is worth its cost | none of the three, at these sizes |
| `pdeclass` | separating the two nine point rules by exactness | both exact to degree 7; the test cannot distinguish them |
| `nonlinearpde` | order against nonlinearity strength | order 2 at every strength; only the wait changes |

## E. Where the sources are wrong or incomplete

Three places where the measurement contradicts a statement that appears in standard treatments,
including the supplied sources.

**"The nine point Laplacian is fourth order."** It is second order in general and sixth on harmonic
functions. Solution 82.3.1 solves with both and measures $2.0021$ and $3.9927$, the second only
after the Mehrstellen correction to the right hand side.

**"The Fisher front travels at $2\sqrt{Dr}$."** That is a lower bound, attained only when the initial
data decays faster than $e^{-a^{*}x}$. Measured at $a = 1$ the front travels at $1.0100$, five times
the quoted value, and for steep data it approaches $2\sqrt{Dr}$ so slowly that at $t = 40$ it is
still $2.3\%$ short.

**"$r \le 1/2$ for the explicit heat equation."** The exact condition is
$r \le 1/(2\cos^{2}(\pi h/2))$, which is always larger. The rule is safe and it is not sharp.

## F. Trefethen and Bau

**Nothing in Part 11 is attributed to Trefethen and Bau.** The book covers numerical linear algebra
and has no material on partial differential equations, and the concept map assigns none of Part 11's
concepts to it. Where Part 11 uses linear algebra it cites the earlier parts that took it from
Trefethen and Bau: lesson 79's non-normality and pseudospectra point at Part 6, lesson 82's
conjugate gradients and preconditioning point at Part 4, and lesson 83's GMRES inner solve points at
Part 4's Krylov chapter.

Lesson 79 is where the debt is largest. The distinction between the spectral radius and the norm,
the Kreiss constant, and the pseudospectral radius are all Trefethen and Bau's material applied to
a PDE operator, and the measurement that follows from them, that the spectral radius is only an
asymptotic statement and the delay before it takes hold is what non-normality really costs, is the
part of the theory that a purely spectral treatment would miss.

## G. What is not here

Three things a full treatment would include and this part does not.

**Finite volume methods are absent.** They are not in the source chapters, and they need
conservation form and flux limiters, which is a subject of its own. The conservation checking in
solution 82.3.2 is the nearest this part comes to the idea.

**Spectral methods appear only as a tool.** The Allen-Cahn solver in solution 83.3.5 uses an FFT for
the diffusion half because the domain is periodic and it is the natural choice there, but the
accuracy and aliasing theory that makes spectral methods a method rather than a convenience belongs
with Part 8's transforms.

**Unstructured meshing is one exercise rather than library code.** Solution 82.3.5 builds an L
shaped domain and its triangulation by hand to measure the reentrant corner exponent, and there is
no mesh generator in `nalib`. The reason is scope: mesh generation is computational geometry, not
numerical analysis, and the exercise reaches the result without it.

## H. Reproducibility

Every lesson fixes `rng = np.random.default_rng(42)` in the standard preamble, and every function in
the seven new modules that uses randomness takes a `seed` argument defaulting to 42. Re-running
`verification/run_all.py` reproduces every number in this document.
