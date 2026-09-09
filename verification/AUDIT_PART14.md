# Audit: Part 14, Numerical Analysis in Machine Learning and AI

A part-level audit. [FINAL_AUDIT.md](FINAL_AUDIT.md) carries the current overall position and
links to every part-level audit, so that list is maintained in one place rather than repeated
here and going stale each time a part lands.

Everything below is measured from the repository by [`run_all.py`](run_all.py). No number in
this document was typed in by hand.

---

## Verdict

**Part 14 passes.** All five lessons exist, all are executed, tested, numerically verified and
fully solved, with no placeholders, no unpaired files and no hardcoded dimensions.

| Check | Result |
|---|---|
| Lessons written | **5 of 5** |
| Notebooks executed end to end | **98 of 98 pass** (every notebook in the repository) |
| Automated tests | **9606 of 9606 pass** (256 new) |
| Assertions inside the Part 14 lessons | **134, all passing** |
| Concepts covered from the sources | **17 new, 514 of 514 total** |
| Concepts outstanding in Part 14 | **0** |
| Worked exercise solutions | **105**, one for every exercise |
| Generality scan | **0 candidates across the repository** |
| Mangled escape scan | **0 control characters across the repository** |

---

## A. Curriculum completeness

Part 14 was planned as 5 lessons. **All five exist.**

| Lesson | Title | Figures | Assertions | Words | Exercises |
|---|---|---:|---:|---:|---:|
| 94 | Numerical Linear Algebra in Machine Learning | 1 | 30 | 3439 | 21 |
| 95 | Optimization for Machine Learning | 1 | 27 | 3386 | 21 |
| 96 | Floating Point in Deep Learning | 1 | 24 | 2558 | 21 |
| 97 | Automatic Differentiation | 1 | 28 | 2590 | 21 |
| 98 | Scientific Machine Learning and Inverse Problems | 1 | 25 | 3691 | 21 |
| **total** | | **5** | **134** | **15664** | **105** |

**Nothing planned was dropped.**

Five new library modules back the part:

| Module | Lines | Public functions | Test lines | Tests |
|---|---:|---:|---:|---:|
| `mllinalg` | 893 | 33 | 446 | 60 |
| `mlopt` | 984 | 25 | 375 | 49 |
| `mixedprecision` | 600 | 21 | 324 | 40 |
| `autodiff` | 955 | 28 | 353 | 45 |
| `sciml` | 1151 | 36 | 558 | 62 |
| **total** | **4583** | **143** | **2056** | **256** |

The solutions are [`part14_ml_and_ai_connections.md`](../solutions/part14_ml_and_ai_connections.md):
4853 lines, 105 solutions, 42 executable code blocks, all of which run.

## B. Source completeness

Part 14 covers **17 of 17** concepts attributed to it, so the part is
marked complete in [`coverage_report.md`](coverage_report.md).

| Source | Chapter | Covered in Part 14 | Covered repository wide |
|---|---|---|---|
| Supplementary | 17 topics no source covers | 17 of 17 | 17 of 17 |

The supplementary concepts are:

- PCA as a truncated SVD, whitening, lesson 94
- Ridge regression as regularized least squares, lesson 94
- Low-rank adaptation as a low-rank matrix factorization, lesson 94
- Gradient descent as an ODE discretization, lesson 95
- Sharp and flat minima, batch size and generalization, lesson 95
- Natural gradient and Gauss-Newton in machine learning, lesson 95
- float16 and bfloat16, exponent and mantissa trade-offs, lesson 96
- The logsumexp trick and numerically stable softmax, lesson 96
- Loss scaling and higher-precision accumulation, lesson 96
- Non-determinism from non-associative floating point reduction, lesson 96
- Forward mode automatic differentiation with dual numbers, lesson 97
- Reverse mode automatic differentiation with a tape, lesson 97
- Cost of automatic differentiation versus finite differences, lesson 97
- Inverse problems and regularization, lesson 98
- Physics-informed neural networks viewed as collocation, lesson 98
- Neural ODEs, lesson 98
- Krylov and low-rank methods inside large-scale learning, lesson 98

**Every concept in this part is supplementary**, because none of the three supplied sources reaches
this material at all. That is the only part of the course for which the whole concept list is
supplementary, and section E says why it is here anyway.

## C. What the measurements changed

Every claim in the five lessons was checked against a measurement, and where the two
disagreed the question asked was which of them was wrong. Below are the 27 cases
where the answer turned out to be the code or the claim rather than the test, in 4
groups: results the measurement contradicted, quantities that were the wrong quantity, fits taken
where the signal was not, and exercises whose expected answer did not survive being run.

This part has more of these than any other, and the reason is structural. Most of its exercises ask
whether a widely repeated statement about machine learning is true, and a claim that has been
repeated often enough to become a slogan has usually lost its conditions along the way.

### Results the measurement contradicted

**"Never form the covariance matrix" is too broad, and the lesson's own measurement is what narrows
it.** Lesson 94's section 3 fitted the error in the **smallest** principal value at a slope of
$2.019$ against the SVD route's $0.945$, which is the classical result and holds. Solution 3.1 then
measured the **leading** components, which is what PCA actually uses, and found the covariance route
as accurate as the SVD: $4\times10^{-14}$ even at a data condition number of $10^{10}$, where its
error on the smallest component is $10^{-2}$. Squaring the spectrum destroys the tail and leaves the
head alone. Both measurements are correct and they are about different quantities, so the honest rule
is: never form it if you need the small directions, which whitening, ridge and every condition
estimate do; for the leading components of a PCA it is fine.

**And the covariance route is genuinely faster, not a habit.** Solution 4.1 measured it at $5$ to $7$
times the SVD route on tall thin data. Combined with the accuracy finding above, the case for it is
substantially stronger than lesson 94's section 3 suggested on its own, and section 3's framing has
been left as it is because the number it reports is right; solution 3.1 supplies the missing half.

**Nesterov momentum is a worse ODE integrator than heavy ball.** Solution 3.1 of lesson 95 fitted
both against the damped oscillator they discretize and found orders of $0.9995$ and $0.9852$ with
Nesterov's error a **factor of $2.3$ larger** at every step size. That is not a contradiction of
Nesterov's reputation, it is a separation of two properties: the $O(1/k^2)$ rate is about how fast
$f(x_k)$ falls with a $k$-dependent momentum, and says nothing about tracking a continuous curve.

**Sharpness-aware minimization never changed basins.** Solution 3.4 of lesson 95 ran it from the
sharp minimum at four perturbation radii and measured the fraction ending in the flat well at
$0.000$ every time. What it does do is measurable and much smaller: the curvature at the endpoint
falls monotonically from $27.24$ to $21.35$, a reduction of $22$ per cent **within** the basin it
started in. SAM perturbs along the gradient without adding energy, so it biases the endpoint towards
lower curvature and cannot carry a run over a barrier.

**The thermal-activation picture of noise-driven escape is rejected by its own fit.** Solution 4.3 of
lesson 95 measured the log escape rate against $1/h$ and fitted a slope of $-4.68$ where Kramers
predicts $-2\Delta/\sigma^2 = -0.745$, a factor of $6.3$. The escape fractions, $0.000$, $0.000$,
$0.017$, $0.100$, $0.983$, are a threshold with a soft edge rather than a smooth exponential. The
noise is not carrying a resting iterate over a barrier; it is perturbing an already growing
oscillation into crossing one.

**Compensated summation in `float16` does not stagnate, it overflows.** Solution 3.4 of lesson 96
expected Kahan to move the stagnation point from $1/u = 2048$ towards $1/u^2 = 4.2\times10^{6}$, and
it does, past the end of the sweep. What ends the `float16` run at $65536$ terms is that
$65536 > 65504$, the format's largest value. So compensation converts a precision failure into a
range failure, which is the same trade the two 16 bit formats make between them, appearing here
inside one of them.

**Adam's second moment is not a curvature estimate.** Solution 3.2 of lesson 95 set out to measure
how much of the condition number it removes and found the premise wrong: on a heteroscedastic model,
where the Fisher and Gauss-Newton matrices stop coinciding, the running average of $g^2$ correlates
with the minibatch **noise** scale at $0.991$ and with $\operatorname{diag}(H)$ at $0.952$. It is a
normalizer, not a preconditioner, and on a deterministic quadratic it degenerates entirely because
$g \to 0$.

**The two Tikhonov penalty operators nearly agree, and the reason is measurable.** Lesson 98's
section 6 expected a smoothness penalty to beat an identity penalty on a smooth answer, and measured
a gain of only $1.037$. The explanation is in the operator: the blur's own right singular vectors are
ordered by roughness over a range of $204$, so shrinking the directions with small $\sigma_i$ **is
already** shrinking the rough ones, and the identity penalty is a smoothness prior in disguise for
this operator.

**Incremental PCA's drift grows with the number of components kept.** Solution 3.5 of lesson 94
measured $2.5$ degrees at $r = 3$ and $14.6$ at $r = 12$, the opposite of the guess that keeping more
would help. The measurement is the largest principal angle between two $r$-dimensional subspaces, so
adding a component adds the direction estimated worst, and that direction sets the angle.

**The crossover where a physics-informed network starts to win is at four dimensions, not past
six.** Lesson 98's solution 4.2 first answered the question by counting coefficients, compared a
network against a **tensor product** basis, and concluded the crossover must lie past $d = 6$. The
comparison is against the wrong classical method. Section 9 now runs it against a Smolyak sparse
grid, whose size is $C(\text{level}+d, d)$ rather than $m^d$, and measures the crossover at $d = 4$
at two different parameter budgets. The counting argument was wrong twice in opposite directions: a
sparse grid survives far past where a tensor product dies, and it still loses earlier than the
tensor count predicted.

**The two `float8` splits differ in range by $239$, not by $16$.** Solution 3.1 of lesson 96 computed
$240$ and $5.7\times10^{4}$ for E4M3 and E5M2 from the exponent widths, so the first draft's estimate
of "about a sixteenth" was wrong by an order of magnitude. Both carry under one decimal digit, $0.90$
and $0.60$, which is the number worth remembering.

### Quantities that were the wrong quantity

**A log-log growth exponent for the blur's condition number.** The first version of lesson 98's
section 3 fitted $\kappa$ against the grid size and got $14.27$, which is not a rate: $\kappa$
saturates at $1/\varepsilon$ by $64$ points, so the fit was measuring where the saturation sat in
the window. Replaced with the quantity that stays meaningful, the count of singular values above
$\sigma_1\varepsilon$, which gives the sharper statement that doubling the grid from $128$ to $256$
adds $128$ unknowns and $3$ usable directions.

**A correlation test between two proportional matrices.** Solution 3.2 of lesson 95 first compared
Adam's second moment against $\operatorname{diag}(H)$ and against the noise scale on a correctly
specified least squares model, and measured $0.995$ and $0.986$. Both are high because for that model
the noise covariance **is** proportional to the Hessian, which is exercise 2.4's Fisher equals
Gauss-Newton identity. The test could not separate the hypotheses, which is a reason to change the
test rather than to report the number.

**A timing comparison between a Python loop and a numpy reduction.** Solution 3.5 of lesson 96
compared fixed-block-count summation against `np.sum` and found a factor of a thousand, which is the
cost of Python and not the cost of determinism. The report now says so; the accuracy column in the
same table is measurable and is reported instead.

**A stagnation test that only suits uncompensated summation.** Solution 3.4 of lesson 96 first looked
for the point where the running total stops changing, which is correct for plain summation and wrong
for Kahan: a compensated sum is allowed to leave the total unchanged for a step, because the lost
part goes into the compensation and comes back later. That test reported Kahan stagnating at exactly
$2048$, the same as plain, contradicting the table three lines above it. Sweeping the length and
measuring the error is the test that answers the question.

**A preconditioning experiment on a problem that was already preconditioned.** Solution 4.2 of lesson
95 set out to move the linear scaling rule's breaking point by preconditioning, and measured no
change, because `least_squares_model` normalizes its design so that $L$ is exactly $1$. Rescaling by
an explicit factor is the same test without the accident, and it gives ratios of $1.28$, $1.28$,
$1.28$ and $1.44$ between the measured break and the predicted one.

### Fits taken where the signal was not

**The $\sqrt\varepsilon$ rule for the finite difference floor is a scale, not a number.** Lesson 97's
section 3 measured the best forward difference at $1.9\times10^{-10}$, which is $0.0125$ of
$\sqrt\varepsilon$, a factor of $80$. The floor is $2\sqrt{\varepsilon|f|\,|f''|}/|f'|$ and the
quoted rule sets all three constants to one. The **step**, by contrast, is predicted well: measured
$2.15\times10^{-8}$ against a predicted $2.12\times10^{-8}$, a ratio of $1.017$. So the location of
the minimum survives dropping the constants and its value does not, and the lesson now reports both.

**A "constant" exponent for pairwise summation.** Lesson 96's section 7 fitted the pairwise spread
against the array length at $0.109$ and the theory says $O(u\log n)$, which is not constant. The
local log-log slope of $\log n$ over $10^3$ to $10^6$ is $0.14$ to $0.07$, and the random-sign
refinement halves that, so $0.109$ sits between the two predictions rather than contradicting either.

**A measured checkpointing minimum compared against a formula that counts differently.** Solution 3.3
of lesson 97 measured $82$ and $41$ against predicted minima of $78.4$ and $34.6$, both a little
high, because the measured count includes the stored checkpoint list as well as the live tape. The
ratio between the two schemes, $2.0$ measured against $2.3$ predicted, is the quantity the formula
actually settles.

**A semiconvergence run cut off before the turn.** Solution 3.3 of lesson 98 ran conjugate gradient
on a two-dimensional deblurring problem for $300$ iterations and reported the error still falling,
which would have licensed the wrong conclusion that semiconvergence does not happen there. It does:
the best iterate is number $637$ and the error has risen by $3.0$ by iteration $4000$. How long the
turn takes is a property of the spectrum, not a universal number.

### Solutions where the exercise's expectation did not survive

**Lesson 94, exercise 4.1** asked where the covariance route wins on cost, expecting the answer to be
"nowhere that matters". It wins by $5$ to $7$ times on tall thin data, and the Gram route wins by
three orders of magnitude on short wide data.

**Lesson 94, exercise 3.5** asked for the drift of incremental PCA, expecting it to fall as more
components are kept. It rises, by a factor of $5.8$ from $r = 3$ to $r = 12$.

**Lesson 95, exercise 3.2** asked how much of the condition number Adam's second moment removes. The
second moment is not estimating the condition number, so the exercise is answered by saying which
matrix it is estimating instead.

**Lesson 95, exercise 4.1** asked whether the largest stable learning rate moves during training, and
the first run, at $0.98$ of the limit, converged to a point with a swing of $5\times10^{-15}$. Being
near the limit is not enough: the amplification is $|1 - 0.98\cdot2| = 0.96 < 1$ and the transient
decays. Only a step **past** the limit produces the period two orbit that is the edge of stability,
and there the run moves itself out to lower curvature until the ratio falls back to $0.960$.

**Lesson 95, exercise 4.3** asked for a fit against the Kramers escape rate, and the fit rejects it.

**Lesson 96, exercise 3.4** asked how far Kahan moves the stagnation point in `float16`, and the
answer is past the format's largest value, so the run overflows before it stagnates.

**Lesson 98, exercise 3.3** asked for a matrix-free two-dimensional deblurring, expecting
semiconvergence to appear as it did in one dimension. It appears $637$ iterations in rather than $60$.

## D. Negative results, reported as such

Part 14 exists to check widely repeated claims, so most of its findings are of this kind. They are
listed here rather than buried in the lessons.

**A physics-informed network loses to a Chebyshev basis by $82000$** on a one-dimensional linear
boundary value problem with a smooth solution, at $1.6\times10^{-11}$ against $1.3\times10^{-3}$, and
the linear basis needs one solve against $111000$ function evaluations. Training the nonlinear basis
is worth a real factor of $10.7$ over the same basis left at its random values, which is the control
that separates what the nonlinearity contributes from what the training does, and most published
comparisons do not run it.

**Cost is not why second order methods are rare.** On flops a Newton step beats gradient descent by
$560$ at $5$ parameters and by $15.3$ at $200$, and the extrapolated crossover, $3148$ parameters,
matches the iteration count gradient descent needed, $3405$, to $7.5$ per cent. The reason is the
noise: at a batch of $8$ with $20$ parameters the sampled curvature has rank $8$ and the run
diverges, and at $32$ Newton is still $6.43$ times worse than plain descent.

**Minibatch noise does nothing outside a window the step size opens.** At a step below the sharp
minimum's stability limit, every batch size from $1$ to $400$ leaves the sharp well in $0$ per cent
of runs. The popular claim that stochastic gradient descent finds flat minima is true here and only
inside a window between two thresholds that a deterministic analysis locates.

**`bfloat16` is eight times worse arithmetic and the right choice.** It carries $2.1$ decimal digits
against `float16`'s $3.0$, and it is the default for training because underflow is fatal and
imprecision is not: at a gradient scale of $10^{-8}$, $99.8$ per cent of a `float16` gradient is
exactly zero and none of a `bfloat16` one is.

**The regularization parameter is much harder to find than it is worth finding.** The L-curve corner
sits $18.7$ times away from the oracle penalty and $1.087$ times away from the oracle error, because
the error curve is nearly flat near its minimum.

**A rank 8 adapter captures $21.07$ per cent of a random $128 \times 128$ matrix**, which is more
than three times the $6.25$ per cent a flat spectrum would give, because even a Gaussian matrix has a
decaying spectrum. The low-rank claim survives only in the comparison against a trained update, which
the adapter captures at $100$ per cent.

**The exact rank bound on a trained update stops binding almost immediately.** After $200$ steps at
batch $8$ the bound $kB = 1600$ exceeds every dimension and both layers' updates are full rank. What
makes an adapter work at that point is that the update's **spectrum** decays, which is an empirical
property rather than the exact theorem of the first few steps.

**A trained basis does beat a sparse grid, from four dimensions on, and the margin is not the
story.** The classical basis wins below the crossover by $7.7\times10^{10}$ and the network wins
above it by $5.06$. Both are true; they are ten orders of magnitude apart. And the crossover is the
grid collapsing rather than the network improving: the errors grow like $10^{3.0d}$ and $10^{0.5d}$,
so the network is not becoming good in high dimensions, it is becoming bad more slowly.

**Past that crossover neither method is usable.** The worst error there is $1.44$ against a
threshold of $0.1$, and a relative error above $1$ is worse than answering zero everywhere. At four
dimensions and above, at these budgets, the winner is the less bad of two answers nobody would use.

**Training the hidden layer is worth between $0.02$ and $15$, and the sign depends on the
dimension.** In one and two dimensions a frozen random basis with a least squares output layer
reaches $10^{-13}$ and training it with Adam makes it worse by up to $10^{10}$, because Adam does
not find the optimum a direct solve hands over. From three dimensions on, training earns up to
$14.6$. Training helps exactly where solving stops being possible.

**Nesterov, sharpness-aware minimization, Kramers escape and Adam's second moment** each turned out
to do something other than what the exercise expected, and are listed in section C above.

## E. Where the sources are wrong or incomplete

None of the three supplied sources covers any of this part. Sauer stops at ordinary differential
equations, Gupta at partial differential equations, and Trefethen and Bau at the singular value
decomposition and iterative methods. Every concept in Part 14 is marked supplementary for that
reason.

That is not a criticism of the sources. It is a statement about when they were written and about
what this part is for: the material here is the point at which a numerical analysis course meets a
literature that mostly does not cite it, and the value of the part is in showing that the two are
about the same objects.

The one place where the wider literature is measurably wrong, rather than merely silent, is the
collection of slogans checked in section D. Four of them survive with conditions attached and three
do not survive at all.

## F. Trefethen and Bau

Trefethen and Bau's lecture 4 defines the singular value decomposition, lecture 5 gives its
properties, and lecture 11 gives least squares. Part 14 uses all three as its foundation: PCA is
lecture 4 applied to a centred matrix, ridge is lecture 11 with a penalty, and the low-rank adapter
is lecture 5's Eckart-Young theorem.

The book's own warning about the normal equations, in lecture 11 and again in lecture 18, is the
warning this part restates for the covariance matrix. Solution 3.1 of lesson 94 refines it: the
warning is about the small singular directions, and the leading ones survive squaring the spectrum.
Trefethen and Bau do not say otherwise, and the refinement is a measurement they did not need to make
for their purposes.

Nothing in this part is attributed to Trefethen and Bau that is not in it. The randomized range
finder of lesson 41, used again here in solution 3.1, postdates the book and is marked supplementary
where it appears.

## G. What is not here

**No neural network training framework.** Every network in this part is a linear layer or a one
hidden layer collocation basis, written out in NumPy. The measurements are about numerical properties
that do not depend on scale, and a framework would have hidden exactly the arithmetic being measured.

**No GPU, so no measured non-determinism.** Lesson 96's section 8 reproduces the mechanism by
chunking a reduction in software and measuring five distinct answers from six block counts. The
mechanism is the whole content; the hardware would add magnitudes and not understanding.

**No `float8` arithmetic, only its format table.** Solution 3.1 of lesson 96 computes the two splits'
properties from their exponent widths and stops there, because simulating the arithmetic would need
the same machinery as `bfloat16` for a format whose accumulation is never done in-format anyway.

**No transformer, no attention, no convolution.** Each would add vocabulary and no numerical content
beyond what a dense layer already shows. The one place where a specific architecture would change an
answer is the rank bound of lesson 94's section 11, and that is stated in terms of the batch size and
the layer shape, which is architecture independent.

**One manufactured solution in the dimension sweep.** Lesson 98's section 9 runs the high
dimensional comparison on $\prod_i \sin(\pi x_i)$, which is smooth and separable, and that is the
case a sparse grid is best at. A solution with a sharp front, or with genuine interaction between
many coordinates, would move the crossover, most likely downwards. The measurement is written so
that substituting a different `poisson` and rerunning is the whole of the work, and it has not been
rerun that way here.

**No dimension above eight.** The sweep stops at six because past that neither method is usable at
any budget a laptop can hold, so the comparison stops being between two answers and becomes a
comparison between two failures. Section D says so with the number.

## H. Reproducibility

Every number in Part 14 comes from code in this repository, run with a fixed seed. The five lessons
build and execute from their sources with `python _planning/lessonbuild.py 94 95 96 97 98`, the five
test modules run with `python -m pytest tests/`, and the whole repository is checked by
`python verification/run_all.py`.

The solutions file executes end to end: `python` over its 42 code blocks, in order,
sharing one namespace, with no failures. Timings are the only numbers in this part that are machine
dependent, and the two tables that contain them say so.
