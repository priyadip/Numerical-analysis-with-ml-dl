# 98. Scientific Machine Learning and Inverse Problems

**Part 14: Numerical Analysis in Machine Learning and AI**

## Learning objectives

By the end of this lesson you will be able to:

1. Say what makes a problem ill-posed, and why refining its discretization makes it worse.
2. Use the discrete Picard condition to decide how much of a dataset is information.
3. Choose a regularization method and its parameter, including the one that is an iteration count.
4. Recognise a physics-informed network as collocation, and set up the comparison that tells you
   whether it is worth using.
5. Explain the adjoint method as implicit differentiation, and say what it trades.

## Prerequisites

Lesson 33 and lesson 94 (least squares, ridge, the SVD filter). Lesson 41 (truncation). Lesson 26
(conjugate gradient). Lesson 76 (collocation and the weak form). Lesson 71 (Runge-Kutta). Lesson 97
(both modes of automatic differentiation, and implicit differentiation).

---

## 1. Where this part ends up

Part 14 has taken four pieces of a machine learning system and found the numerical analysis inside
each. This lesson closes the loop by going the other way: taking machine learning methods aimed at
scientific problems, and asking what they are and what they are worth against the methods this
course already has.

| What it is called | What it is | Lesson |
|---|---|---|
| Inverse problem | Least squares with a smoothing operator, so $\kappa$ is enormous | 33 |
| Tikhonov regularization | Ridge, with a choice of penalty operator | 94 |
| Early stopping | Regularization whose parameter is the iteration count | 26 |
| Physics-informed neural network | Collocation with a nonlinear basis and a nonconvex fit | 76 |
| Neural ODE | An initial value problem with a fitted right-hand side | 67 |
| Adjoint method | Implicit differentiation of the ODE, at constant memory | 97 |

The point of the table is not that the new names are unnecessary. It is that once the object is
named correctly, the question of whether the method helps becomes a measurement.

## 2. An inverse problem smooths, so its inverse does not exist

The model problem is a Fredholm equation of the first kind,

$$
\int_0^1 k(s - t)\, x(t)\, dt = b(s) ,
$$

with a Gaussian kernel $k$. Discretize it and you get $Ax = b$ with $A$ symmetric and positive.
Deblurring, computed tomography, and heat conduction run backwards are all this equation.

The trouble is that integrating against a smooth kernel destroys high frequencies, so $A$ maps a
large ball to a very flat pancake. Its singular values decay **geometrically**, and the inverse
multiplies by their reciprocals.

```python
from nalib import sciml

problem = sciml.blur(64)
values = np.linalg.svd(problem["matrix"], compute_uv=False)
print(f"a {problem['size']} by {problem['size']} blur with width {problem['width']}")
print(f"{'i':>5}{'sigma_i':>15}{'ratio to the last':>20}")
for i in (0, 1, 2, 5, 10, 20, 40, 63):
    ratio = "" if i == 0 else f"{values[i] / values[i - 1]:.6f}"
    print(f"{i:>5}{values[i]:>15.4e}{ratio:>20}")
print(f"\ncondition number {values[0] / values[-1]:.3e}")

assert values[0] / values[-1] > 1e10
```

The ratio between consecutive singular values settles below $1$ and stays there, so
$\sigma_i \sim c\,r^i$. At $64$ points the condition number is already past $10^{17}$.

**This is not a badly written program.** The continuous problem has no bounded inverse, and every
discretization inherits that. The question is never how to solve it accurately; it is how much of it
to solve.

## 3. Refining the grid makes it worse

Here is the property that separates an inverse problem from everything else in this course. In Parts
9 through 11, a finer grid meant a better answer. Here it does not.

```python
from nalib import sciml

out = sciml.refining_the_grid_makes_an_inverse_problem_worse()
print(f"{'grid':>7}{'kappa(blur)':>15}{'kappa(second difference)':>27}"
      f"{'decay per index':>18}{'usable in double':>19}")
for row in out["rows"]:
    print(f"{row['size']:>7}{row['blur_condition']:>15.3e}"
          f"{row['difference_condition']:>27.3e}{row['decay_per_index']:>18.6f}"
          f"{row['usable_at_double']:>19}")
print(f"\nthe second difference operator grows like n to the {out['difference_growth']:.4f}")
print(f"the blur passes 1/eps = {out['double_precision_ceiling']:.2e} at "
      f"{out['saturates_at']} points")
print(f"the last doubling adds {out['grid_gain_from_the_last_doubling']} unknowns and "
      f"{out['usable_gain_from_the_last_doubling']} usable singular values")

assert out["the_blur_exhausts_double_precision"]
assert out["the_extra_grid_points_are_wasted"]
assert out["the_difference_operator_grows_like_n_squared"]
```

The second difference operator of lesson 77 grows like $n^{1.975}$, which is $n^2$ and is fine: at
$256$ points its condition number is $26768$ and it loses four digits.

The blur passes $1/\varepsilon$ at $64$ points. And the last column is the sharpest statement in
this lesson: **going from $128$ to $256$ points adds $128$ unknowns and $3$ usable singular
values.** Everything past that is below the rounding of the operator itself.

So resolution and information are different things here. A finer grid gives a longer answer vector
and not one bit more of an answer.

## 4. How much of the data is information

The **discrete Picard condition** answers that. Write $b$ in the left singular basis and compare
$|u_i^{\mathsf T}b|$ against $\sigma_i$. The solution is
$x = \sum_i (u_i^{\mathsf T}b/\sigma_i)\,v_i$, so a component is usable only while the numerator
falls faster than the denominator.

Noise does not decay in $i$ at all. It is white, so its coefficients sit at a constant level, and
where that level crosses $\sigma_i$ is where the information stops.

```python
from nalib import sciml

out = sciml.the_picard_condition_locates_the_usable_components()
print(f"{'noise':>10}{'usable of 128':>16}{'noise floor':>15}{'pseudoinverse error':>22}")
for row in out["rows"]:
    print(f"{row['noise']:>10.0e}{row['usable']:>16}{row['noise_floor']:>15.3e}"
          f"{row['pseudoinverse_error']:>22.4e}")

print(f"\n{'i':>5}{'sigma_i':>14}{'|u_i.b|':>14}{'ratio':>14}")
for i in (0, 8, 16, 24, 32, 48, 64, 96):
    print(f"{i:>5}{out['singular_values'][i]:>14.3e}{out['coefficients'][i]:>14.3e}"
          f"{out['coefficients'][i] / out['singular_values'][i]:>14.3e}")

assert out["usable_falls_with_noise"]
assert out["the_pseudoinverse_fails"]
```

**With noise at $10^{-4}$ only $31$ of $128$ components are usable**, and the plain pseudoinverse
answer is wrong by a factor of $5.8\times10^{7}$. At $10^{-2}$ it is $22$ components.

Notice the top row. Even with **no** added noise the count is $61$ rather than $128$, because the
operator is stored in floating point and $\varepsilon$ is a noise level. **There is no such thing as
exact data.**

## 5. Tikhonov is the ridge of lesson 94

The fix is the one lesson 94 already derived: minimize
$\lVert Ax - b\rVert^2 + \lambda\lVert Lx\rVert^2$. With $L = I$ this is ridge regression, filter
factors and all.

```python
from nalib import sciml

out = sciml.tikhonov_is_the_ridge_of_lesson_94()
print(f"the filter form and the solve agree to {out['filter_gap']:.3e}")
print(f"\n{'penalty':>12}{'residual':>14}{'solution norm':>16}{'error':>12}")
for row in out["rows"][::6]:
    print(f"{row['penalty']:>12.2e}{row['residual']:>14.6e}{row['norm']:>16.4f}"
          f"{row['error']:>12.6f}")
print(f"\nL-curve corner at lam = {out['corner_penalty']:.3e}, error {out['corner_error']:.6f}")
print(f"oracle           at lam = {out['oracle_penalty']:.3e}, error {out['oracle_error']:.6f}")
print(f"the penalties differ by {out['penalty_ratio']:.2f} and the errors by "
      f"{out['error_ratio']:.4f}")

assert out["it_is_the_same_filter"]
assert out["the_corner_is_close_to_the_oracle"]
```

**The filter factors match to $8.8\times10^{-12}$**: it is the same computation with a different
name.

The last two lines are the useful result, and they are a relief. The L-curve corner, which uses only
the data, picks a penalty **$18.7$ times** away from the oracle penalty, which uses the answer. But
the errors differ by only $1.087$. **The error curve is nearly flat near its minimum**, so the
penalty is much harder to find than it is worth finding, and any method that gets within an order of
magnitude is good enough.

## 6. The penalty operator is a prior, and this operator already has one

$L = I$ says the answer is small. $L = D_2$ says it is smooth. These are different beliefs, so the
honest test is to run both on answers that do and do not have the property.

```python
from nalib import sciml

out = sciml.a_derivative_penalty_knows_something_the_identity_does_not()
print(f"{'signal':>10}{'best lam (I)':>15}{'error (I)':>13}{'best lam (D2)':>16}"
      f"{'error (D2)':>13}{'gain':>9}{'truncated':>12}")
for row in out["rows"]:
    print(f"{row['signal']:>10}{row['identity_penalty']:>15.3e}{row['identity_error']:>13.6f}"
          f"{row['derivative_penalty']:>16.3e}{row['derivative_error']:>13.6f}"
          f"{row['gain']:>9.4f}{row['truncated_error']:>12.6f}")

print(f"\nroughness ||D2 v_i|| of the blur's own right singular vectors:")
print(f"{'i':>6}{'roughness':>14}")
for index, value in out["roughness_shown"]:
    print(f"{index:>6}{value:>14.4e}")
print(f"\nover a range of {out['roughness_range']:.1f}, rising until index "
      f"{out['roughness_peaks_at']}")

assert out["the_two_penalties_nearly_agree"]
assert out["the_operator_sorts_its_own_directions_by_smoothness"]
```

**The two penalties differ by at most $1.037$**, which is a negative result, and the second table
says why. The blur's own right singular vectors are ordered by roughness, from $172$ to
$3.5\times10^{4}$, so shrinking the directions with small $\sigma_i$ **is already** shrinking the
rough ones. The identity penalty is a smoothness penalty in disguise, for this operator.

That will not be true of every operator, and the measurement is what tells you which case you are
in. The one visible difference is in the right direction: the smoothness penalty gains $1.037$ on a
smooth answer and $1.004$ on a discontinuous one.

## 7. Stopping early is regularization

Conjugate gradient from lesson 26 needs only a matrix-vector product, so it solves $A^{\mathsf T}Ax
= A^{\mathsf T}b$ without forming anything. It also picks up the large singular directions first,
which means its early iterates are already smooth approximations.

Run it to convergence and it finds the least squares solution, which is exactly the answer section 4
showed is useless. So the iteration count itself is a regularization parameter.

```python
from nalib import sciml

out = sciml.stopping_early_is_regularization()
print(f"conjugate gradient on the normal equations, {out['iterations']} iterations")
print(f"{'noise':>10}{'best iteration':>17}{'error there':>14}{'error at the end':>19}"
      f"{'rise':>12}{'best Tikhonov':>16}")
for row in out["rows"]:
    print(f"{row['noise']:>10.0e}{row['best_iteration']:>17}{row['best_error']:>14.6f}"
          f"{row['final_error']:>19.4e}{row['rise']:>12.1f}{row['best_penalty_error']:>16.6f}")

history = out["histories"][1e-4]
print(f"\nthe error history at noise 1e-4:")
print("  " + "  ".join(f"{history[i]:.3f}" for i in (0, 4, 19, 39, 59, 99, 199, 399)))

assert out["it_falls_then_rises"]
assert out["the_minimum_moves_earlier_with_noise"]
assert out["it_is_competitive_with_tikhonov"]
```

**The error falls to $0.1239$ at iteration $60$ and then rises to $14.6$ by iteration $400$**, a
factor of $118$. This is called **semiconvergence**, and it is the reason an ill-posed problem must
never be handed to a solver with a tight tolerance and left alone.

Two more things in the table. **The best iteration moves earlier as the noise grows**, from $98$ at
$10^{-5}$ to $7$ at $10^{-2}$, exactly as the best penalty moves larger. And at its best, conjugate
gradient is $0.9$ per cent **better** than the best Tikhonov solution, so early stopping is not a
cheap approximation to regularization. It is regularization, with a discrete parameter.

## 8. A physics-informed network is collocation

Lesson 76 solved $u'' = f$ by picking a basis, writing the residual at a set of points, and solving
for the coefficients. A physics-informed network does exactly that. The differences are that the
basis is nonlinear in its own parameters, and that the fit is by gradient descent instead of by a
linear solve.

That makes the comparison clear. Solve one problem three ways with the same number of unknowns:

```text
Chebyshev basis, fixed          one linear solve
tanh basis, parameters frozen   one linear solve, nonlinear basis, random placement
tanh basis, trained end to end  a nonconvex fit, which is the network
```

The middle one is the control. Without it, any difference between the first and third could be
attributed to the basis or to the training, and there would be no way to tell.

```python
from nalib import sciml

out = sciml.collocation_beats_the_trained_basis()
print(f"{'Chebyshev terms':>18}{'solves':>9}{'error':>15}")
for row in out["linear"]:
    print(f"{row['terms']:>18}{row['solves']:>9}{row['error']:>15.4e}")
print(f"\n{out['units']} tanh units, frozen at random values, one solve: "
      f"{out['frozen_tanh_error']:.4e}")
print(f"{out['units']} tanh units, trained for {out['steps']} Adam steps  : "
      f"{out['trained_tanh_error']:.4e}")
print(f"\ntraining gains a factor of {out['training_gain']:.2f} over the frozen basis")
print(f"the same size Chebyshev basis gets {out['matched_chebyshev_error']:.4e} in one solve")
print(f"a basis twice the size gets {out['best_chebyshev_error']:.4e}")
print(f"the trained fit used {out['evaluations']} function evaluations")

assert out["training_helps"]
assert out["the_linear_basis_matches_it_in_one_solve"]
assert out["a_bigger_linear_basis_beats_it_outright"]
```

Three findings, and the first one is in the network's favour.

**Training the basis is worth a factor of $10.7$.** The same tanh units left at random values give
$0.0143$; trained, they give $0.00133$. The nonlinearity is doing real work, and a fair account has
to say so.

**A linear basis of the same size matches it in one solve.** Chebyshev with $12$ terms gives
$0.00125$ against the trained network's $0.00133$, and it took one least squares solve against
$111000$ function evaluations.

**A larger linear basis beats it outright.** At $24$ terms the Chebyshev error is
$1.6\times10^{-11}$, better by a factor of $82000$, and the cost is still one solve. The Chebyshev
error falls by $1.2\times10^{10}$ as the terms go from $8$ to $24$, which is the spectral convergence
of lesson 47. **The trained fit has nothing like that**: its accuracy is set by how well the
optimizer did, and there is no knob that improves it by ten orders of magnitude.

So the honest summary is that on a one-dimensional linear problem with a smooth solution, the
physics-informed approach is the wrong tool, and the reasons it is used elsewhere are the ones this
problem does not have: high dimension, where a tensor-product basis is impossible; irregular
geometry; and unknown terms in the equation itself.

## 9. Where a trained basis starts to win

Section 8 measured the case a physics-informed network is not for: one dimension, a linear problem,
a smooth solution, and a Chebyshev basis that can represent it. The network lost by $82000$.

The claim usually made for these methods is about **high dimension**, so this is the comparison that
could reverse it. Solve the same Poisson problem,

$$
\Delta u = f \ \text{ on } [0,1]^d , \qquad u = 0 \text{ on the boundary},
\qquad u^\star(x) = \prod_{i=1}^{d}\sin(\pi x_i) ,
$$

three ways at a **matched parameter budget**, as $d$ grows.

```text
sparse grid      Chebyshev multi-indices with sum(a_i) <= level
                 size C(level+d, d), which is polynomial in d, not m**d

trained tanh     one hidden layer, fitted end to end by Adam on the PDE residual
                 units * (d + 2) parameters, so the count barely grows with d

frozen tanh      the same network, hidden layer left at its random values,
                 output layer by least squares.  This is the control
```

The full tensor product basis is not in the table because it is the thing being avoided: at
$d = 6$ with $6$ points per axis it is $46656$ coefficients, and at $d = 8$ it is $1.7$ million.
A sparse grid is the honest classical competitor, and it is what a numerical analyst would actually
reach for.

One detail that is a numerical constraint rather than a modelling choice. The per-axis degree is
capped at $16$, because a Chebyshev second derivative grows like $n^4$, so past that the collocation
matrix is conditioned out of double precision and the failure measured would be lesson 33's rather
than the dimension's.

```python
from nalib import sciml

out = sciml.where_a_trained_basis_starts_to_win()
for budget in sorted(out["crossovers"]):
    print(f"\nbudget {budget} parameters, {out['steps']} training steps")
    print(f"{'d':>4}{'level':>7}{'basis':>8}{'sparse grid':>14}{'units':>7}"
          f"{'trained net':>14}{'frozen net':>13}{'grid / net':>13}")
    for row in out["rows"]:
        if row["budget"] != budget:
            continue
        print(f"{row['dimension']:>4}{row['level']:>7}{row['grid_size']:>8}"
              f"{row['grid_error']:>14.3e}{row['units']:>7}{row['trained_error']:>14.3e}"
              f"{row['frozen_error']:>13.3e}{row['ratio']:>13.3g}")
    print(f"  crossover at d = {out['crossovers'][budget]}, "
          f"grid error grows like 10^({out['slopes'][budget]['grid']:.2f} d), "
          f"trained like 10^({out['slopes'][budget]['trained']:.2f} d)")

print(f"\nlargest classical win {out['largest_classical_win']:.3g}, "
      f"largest trained win {out['largest_trained_win']:.3g}")
print(f"worst error past the crossover: {out['worst_error_past_the_crossover']:.3g}, "
      f"against a usable threshold of {out['usable_threshold']}")

assert out["the_classical_basis_wins_in_low_dimensions"]
assert out["the_trained_basis_wins_somewhere"]
assert out["the_crossover_agrees_across_budgets"]
assert out["neither_is_usable_past_the_crossover"]
```

**The crossover is at $d = 4$, and it is the same at both budgets.** That is the answer to the
question section 8 could not ask, and it is worth four separate readings.

**The classical basis wins below it by an amount nothing else in this comparison approaches.** At
$d = 1$ and $d = 2$ the sparse grid is at $10^{-13}$ and the network at $10^{-3}$, a ratio of
$7.7\times10^{10}$. The largest win in the other direction is $5.06$. **So "the network wins in high
dimensions" is true, and the margin is ten orders of magnitude smaller than the margin it loses by in
low ones.**

**The crossover happens because the grid collapses, not because the network improves.** The fitted
growth of the error with the dimension is $10^{3.0d}$ for the sparse grid and $10^{0.5d}$ for the
network, a factor of six in the exponent. The network is not getting good in high dimensions, it is
getting bad more slowly. Read the level column to see the mechanism: at a fixed budget the sparse
level falls from $16$ to $4$ as $d$ goes from $1$ to $6$, because $C(\text{level}+d, d)$ grows in
both arguments.

**And past the crossover neither method works.** The worst error there is $1.44$, and a relative
error above $1$ means worse than answering zero everywhere. At $d \ge 4$ the winner of this
comparison is the less bad of two answers you would not use. Getting a usable answer at four
dimensions and above needs a budget larger than $400$ for either method, which is the honest form of
what "high dimensional problems are hard" means.

**The frozen column is the control, and it says training is worth between $0.02$ and $15$.** In one
and two dimensions the random basis with a least squares output layer reaches $10^{-13}$ and
training it with Adam makes it **worse**, by up to $10^{10}$, because Adam does not find the least
squares optimum a direct solve hands over. From three dimensions on, where the least squares problem
is no longer easy, training earns a factor of up to $14.6$. **Training the basis helps exactly where
solving for it stops being possible.**

## 10. A neural ODE, and the adjoint method

A neural ODE replaces a stack of layers with

$$
\frac{dz}{dt} = f(z, \theta) , \qquad z(0) = x ,
$$

integrated to $t = 1$. That is lesson 67's initial value problem with a fitted right-hand side, and
training it needs $\partial L/\partial\theta$ where $L$ depends on $z(1)$.

Lesson 97 gave the two routes, and here they have names.

**Discretize then optimize.** Differentiate the solver's arithmetic. Exact for the discrete problem,
and the tape holds every intermediate state.

**Optimize then discretize.** Differentiate the equation. The adjoint $a = \partial L/\partial z$
satisfies

$$
\frac{da}{dt} = -a^{\mathsf T}\frac{\partial f}{\partial z} ,
$$

integrated backwards from $a(1) = \partial L/\partial z(1)$, and

$$
\frac{\partial L}{\partial\theta} = -\int_0^1 a^{\mathsf T}\frac{\partial f}{\partial\theta}\,dt .
$$

Memory is one state and one adjoint, whatever the step count. This is section 10 of lesson 97, the
implicit function theorem, applied to a differential equation.

```python
from nalib import sciml

out = sciml.the_adjoint_gives_the_same_gradient_at_constant_memory()
print(f"{'steps':>8}{'h':>12}{'gap between the two gradients':>32}"
      f"{'adjoint memory':>17}{'unrolled memory':>18}{'ratio':>9}")
for row in out["rows"]:
    print(f"{row['steps']:>8}{row['step']:>12.6f}{row['gap']:>32.6e}"
          f"{row['adjoint_memory']:>17}{row['unrolled_memory']:>18}"
          f"{row['memory_ratio']:>9.1f}")
print(f"\nthe gap falls like h to the {out['order']:.4f}")

assert out["they_converge"]
assert out["adjoint_memory_is_constant"]
assert out["unrolled_memory_grows"]
```

**The two gradients are not the same number.** They differ by $2.2\times10^{-2}$ at $16$ steps and
$6.8\times10^{-4}$ at $512$, falling like $h^{0.998}$.

That is not a bug in either. They are gradients of **different functions**. Unrolling gives the
exact gradient of the discrete problem you are actually solving; the adjoint gives a discretization
of the exact gradient of the continuous problem you meant to solve. Which one you want depends on
what you are optimizing, and the usual answer is that you want the discrete one, because that is the
function whose value you are measuring.

**The adjoint's memory is $18$ numbers at every step count**, while unrolling needs $6156$ at $512$
steps, a ratio of $342$ that keeps growing. That is what the method is for, and it is the same trade
as lesson 97's checkpointing, at the extreme.

## 11. The picture

```python
from nalib import sciml

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

problem = sciml.blur(96)
truth = sciml.test_signal(problem["grid"])
made = sciml.noisy_data(problem, truth, level=1e-4, seed=42)
axes[0].plot(problem["grid"], truth, "k-", lw=2.0, label="the answer")
axes[0].plot(problem["grid"], made["data"], "-", lw=1.0, label="the data")
for penalty, style in ((1e-10, ":"), (5e-7, "-"), (1e-3, "--")):
    axes[0].plot(problem["grid"], sciml.tikhonov(problem, made["data"], penalty)["solution"],
                 style, lw=1.2, label=f"lam = {penalty:.0e}")
axes[0].set_ylim(-1.5, 2.0)
axes[0].set_xlabel("t")
axes[0].set_title("too little, right, too much")
axes[0].legend(fontsize=6)

picard = sciml.the_picard_condition_locates_the_usable_components()
index = np.arange(picard["singular_values"].size)
axes[1].semilogy(index, picard["singular_values"], "-", label="sigma_i")
axes[1].semilogy(index, picard["coefficients"], ".", ms=3, label="|u_i . b|")
axes[1].semilogy(index, picard["coefficients"] / picard["singular_values"], ".", ms=3,
                 label="their ratio")
axes[1].set_ylim(1e-12, 1e12)
axes[1].set_xlabel("index")
axes[1].set_title("the discrete Picard condition")
axes[1].legend(fontsize=7)

stopping = sciml.stopping_early_is_regularization()
for level, style in ((1e-5, "-"), (1e-4, "--"), (1e-3, "-."), (1e-2, ":")):
    history = stopping["histories"][level]
    axes[2].semilogy(np.arange(1, len(history) + 1), history, style,
                     label=f"noise {level:.0e}")
axes[2].set_xlabel("conjugate gradient iteration")
axes[2].set_ylabel("relative error")
axes[2].set_title("semiconvergence")
axes[2].legend(fontsize=7)

fig.tight_layout(); fig.savefig("../figures/98_sciml.png", dpi=110); plt.close(fig)
print("saved ../figures/98_sciml.png")
```

![Tikhonov solutions at three penalties, the discrete Picard plot, and the semiconvergence of conjugate gradient at four noise levels](../figures/98_sciml.png)

The left panel is section 5: too small a penalty gives noise, too large gives a smear, and in between
is the answer. The middle is section 4, with the singular values falling geometrically, the data
coefficients flattening at the noise floor, and their ratio turning upwards exactly where the two
cross. The right is section 7: four U curves, each with its minimum earlier than the last.

## 12. From scratch

Regularized deconvolution is four lines, and choosing the parameter is the rest of the work.

```python
def my_tikhonov(matrix, data, penalty):
    """One stacked least squares problem, exactly as in lesson 94."""
    stacked = np.vstack([matrix, np.sqrt(penalty) * np.eye(matrix.shape[1])])
    padded = np.concatenate([data, np.zeros(matrix.shape[1])])
    return np.linalg.lstsq(stacked, padded, rcond=None)[0]


def my_discrepancy(matrix, data, level, penalties):
    """Morozov's principle: pick the largest penalty whose residual still matches the noise."""
    wanted = level * np.sqrt(data.size)
    for penalty in sorted(penalties, reverse=True):
        solution = my_tikhonov(matrix, data, penalty)
        if np.linalg.norm(matrix @ solution - data) <= wanted:
            return penalty, solution
    return min(penalties), my_tikhonov(matrix, data, min(penalties))


from nalib import sciml

problem = sciml.blur(96)
truth = sciml.test_signal(problem["grid"])
made = sciml.noisy_data(problem, truth, level=1e-4, seed=42)
grid = np.geomspace(1e-12, 1e2, 45)

chosen, solution = my_discrepancy(problem["matrix"], made["data"], made["level"], grid)
mine = float(np.linalg.norm(solution - truth) / np.linalg.norm(truth))
library = sciml.tikhonov(problem, made["data"], chosen)["solution"]

reference = sciml.tikhonov_is_the_ridge_of_lesson_94()
print(f"the discrepancy principle picks lam = {chosen:.3e}, error {mine:.6f}")
print(f"the L-curve picked           lam = {reference['corner_penalty']:.3e}, "
      f"error {reference['corner_error']:.6f}")
print(f"the oracle would pick        lam = {reference['oracle_penalty']:.3e}, "
      f"error {reference['oracle_error']:.6f}")
print(f"\nmy solve agrees with the library to "
      f"{float(np.max(np.abs(solution - library))):.3e}")
print(f"the unregularized answer is off by "
      f"{float(np.linalg.norm(np.linalg.lstsq(problem['matrix'], made['data'], rcond=None)[0] - truth) / np.linalg.norm(truth)):.3e}")

assert np.allclose(solution, library, atol=1e-10)
assert mine < 0.5
```

Three parameter choices, three different penalties, and errors within a factor of each other.
**Every one of them beats the unregularized answer by nine orders of magnitude**, which is the point:
choosing the parameter well matters much less than regularizing at all.

## 13. Exercises

**Level 1, understanding**

1.1 Say what makes a problem ill-posed and give the property of the operator that causes it.

1.2 Explain why refining the grid does not help, in terms of the singular values.

1.3 State the discrete Picard condition and say what it is used for.

1.4 Say what early stopping regularizes and what its parameter is.

1.5 Give the two routes to the gradient of a neural ODE and the property that distinguishes them.

**Level 2, derivation**

2.1 Show that the singular values of a discretized smoothing kernel decay geometrically for an
analytic kernel.

2.2 Derive the Tikhonov filter factors and show that truncation is their limit as the transition
sharpens.

2.3 Derive the adjoint equation from the Lagrangian of the constrained problem.

2.4 Show that the conjugate gradient iterate after $k$ steps is a polynomial filter of the singular
values, and say why the early iterates are smooth.

2.5 Derive the discrepancy principle and say what it needs to know that the L-curve does not.

**Level 3, computational**

3.1 Implement generalized cross-validation and compare its penalty against the L-curve and the
discrepancy principle over a range of noise levels.

3.2 Implement total variation regularization with the proximal gradient method of lesson 90, and
compare it against Tikhonov on the discontinuous test signal.

3.3 Implement a two-dimensional deblurring problem with a Kronecker product operator, and solve it
with conjugate gradient without ever forming the matrix.

3.4 Implement a physics-informed solver for a nonlinear boundary value problem, where the linear
basis needs Newton's method too, and redo section 8's comparison.

3.5 Implement the adjoint method with a checkpointed forward pass and measure the memory against
both extremes.

**Level 4, experimental**

4.1 Measure how the best regularization parameter scales with the noise level, and compare it
against the theoretical rate.

4.2 Section 9 found the crossover at $d = 4$ against a sparse grid. Repeat it against a full tensor
product basis instead, and explain why the crossover moves.

4.3 Measure the gap between the two neural ODE gradients as the solver's order changes, and say
whether a higher order solver closes it faster.

**Level 5, advanced**

5.1 **Why a finer grid does not help.** Explain what a discretization of an ill-posed problem is
converging to, and what the right notion of convergence is.

5.2 **When a physics-informed network is the right choice.** Sections 8 and 9 give one reversal, in
the dimension. Describe the other problem features that reverse it, and say what measurement would
confirm each.

5.3 **Discretize or optimize first.** Given section 10, argue for each choice and say which one a
practitioner should default to and why.

## 14. Key takeaways

- **An inverse problem is least squares with a smoothing operator**, so its singular values decay
  geometrically and its condition number passes $10^{17}$ at $64$ grid points.

- **Refining the grid makes it worse.** From $128$ to $256$ points adds $128$ unknowns and $3$
  usable singular values, while a second difference operator on the same grid grows only like
  $n^{1.975}$.

- **The discrete Picard condition counts the information.** At a noise level of $10^{-4}$, $31$ of
  $128$ components are usable, and the pseudoinverse is wrong by $5.8\times10^{7}$.

- **There is no exact data.** With no added noise the count is $61$ of $128$, because $\varepsilon$
  is a noise level.

- **Tikhonov is ridge**, matching lesson 94's filter factors to $8.8\times10^{-12}$.

- **The penalty is harder to find than it is worth finding.** The L-curve corner is $18.7$ times
  from the oracle penalty and $1.087$ times from the oracle error.

- **The two penalty operators nearly agree**, at a gain of $1.037$, because the blur's own singular
  vectors are already ordered by roughness over a range of $204$.

- **Stopping early is regularization.** The error falls to $0.1239$ at iteration $60$ and rises to
  $14.6$ by $400$, the best iteration moves from $98$ to $7$ as the noise grows, and at its best it
  beats the best Tikhonov solution by $0.9$ per cent.

- **A physics-informed network is collocation**, and training the basis is worth a factor of $10.7$
  over the same basis left at random.

- **A linear basis of the same size matches it in one solve**, $0.00125$ against $0.00133$, and a
  basis twice the size beats it by $82000$ with spectral convergence the trained fit cannot match.

- **The crossover where a trained basis starts to beat a sparse grid is at $d = 4$**, the same at
  both parameter budgets. Below it the classical basis wins by up to $7.7\times10^{10}$; above it
  the network wins by at most $5.06$.

- **The crossover is the grid collapsing, not the network improving.** The errors grow like
  $10^{3.0d}$ and $10^{0.5d}$, and at a fixed budget the sparse level falls from $16$ to $4$.

- **Past the crossover neither method is usable**, at a worst error of $1.44$ against a threshold of
  $0.1$.

- **Training the hidden layer is worth between $0.02$ and $15$.** It makes things worse in one and
  two dimensions, where a direct solve for the output layer already reaches $10^{-13}$, and earns a
  factor of $14.6$ from three dimensions on.

- **The adjoint method costs constant memory**, $18$ numbers against $6156$ at $512$ steps.

- **The two neural ODE gradients are gradients of different functions**, agreeing only as
  $h \to 0$ at a fitted order of $0.998$.

## Where this goes next

Nowhere. This is the last lesson.

What the course has been building is one habit: when a method is proposed, find out what it actually
is, then measure it against the thing it is supposed to beat. Part 14 has done that five times, and
the answers went in both directions. Low-rank adaptation works, and for a reason that is not the one
usually given. Second order methods are rare, and not for the reason usually given. `bfloat16` is
worse arithmetic and the right choice. A physics-informed network really does learn its basis, and
still loses to a Chebyshev polynomial on a problem a Chebyshev polynomial can represent.

And a physics-informed network does beat a sparse grid, from four dimensions on, by a factor of
five, in a regime where neither of them is accurate enough to use.

None of those answers could have been reached by reading. Every one of them is in this repository as
code that runs, and the way to disagree with any of it is to change the code and see what happens.
