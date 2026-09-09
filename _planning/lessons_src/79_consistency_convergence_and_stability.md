# 79. Consistency, Convergence and Stability

**Part 11: Partial Differential Equations**

## Learning objectives

By the end of this lesson you will be able to:

1. Define the three properties separately and measure each one on its own.
2. Compute a truncation constant, not just an order, and check it against a closed form.
3. Run the matrix method and von Neumann analysis and say exactly when they are the same statement.
4. Explain what non-normality costs, with a measurement rather than a warning.
5. Use the Lax equivalence theorem in the only way it can be used: by knowing which two
   combinations it forbids.

## Prerequisites

Lesson 78 (the weighted family, the mesh ratio and the growth factor, used throughout).
Lesson 27 (the spectral radius and why it governs an iteration). Lesson 26 (non-normal matrices
and the gap between eigenvalues and behaviour). Lesson 71 (zero-stability, which is the same idea
for multistep ODE methods).

---

## 1. Three properties, one theorem

**Consistency** is about the exact solution. Put it into the difference formula and see what is
left over. If that residual goes to zero as the grid is refined, the scheme is consistent, and the
rate is its order.

**Stability** is about the scheme alone, with no reference to any solution. If the operator that
advances one step has bounded powers, the scheme is stable.

**Convergence** is about the computed solution. If it approaches the exact one as the grid is
refined, the scheme converges.

Only the third is what you want, and it is the hardest to establish directly. That is what the
**Lax equivalence theorem** is for: for a well posed linear initial value problem and a
**consistent** scheme,

$$
\text{stability} \iff \text{convergence}.
$$

Consistency is a Taylor expansion. Stability is an eigenvalue calculation. Convergence is neither,
and the theorem hands it to you from the other two.

## 2. Consistency, with the constant and not just the order

Expanding the weighted scheme about $(x_j, t_n + k/2)$ and using $u_{tt} = \alpha^2 u_{xxxx}$ on a
solution of the equation collapses the whole family into one expression:

$$
T = \alpha h^2\left[\left(\tfrac12 - \theta\right)r - \tfrac{1}{12}\right]u_{xxxx} + O(h^4).
$$

An order fit would report 2 for every member and stop there. The **constant** is the bracket, and
it is what separates them.

```python
from nalib import parabolic as pb
from nalib import pdestability as ps

out = ps.consistency_of_the_family(ratio=0.4)
print(f"at r = {out['ratio']}")
print(f"{'theta':>9}{'order in h':>13}{'constant':>13}{'predicted':>13}{'agrees':>9}")
for row in out["rows"]:
    print(f"{row['theta']:>9.4f}{row['order_in_h']:>13.3f}{row['constant']:>13.4f}"
          f"{row['predicted_constant']:>13.4f}{str(row['agrees']):>9}")
print(f"\nthe formula predicts every constant: {out['the_formula_predicts_every_constant']}")

assert out["the_formula_predicts_every_constant"]
assert out["every_member_is_consistent"]
```

The measured constants match the closed form to three digits at every $\theta$, which is a far
stronger check on the expansion than a fitted order would be.

Two consequences follow, and the second is not the one most treatments give.

**The bracket vanishes at $\theta = \tfrac12 - \tfrac{1}{12r}$**, and there the scheme is fourth
order. At $r = 0.4$ that is $\theta = 0.2917$, and the measured order jumps to 3.99.

```python
print(f"the cancelling theta at r = {out['ratio']} is {out['cancelling_theta']:.4f}")
print(f"its measured order is {out['order_at_the_cancelling_theta']:.3f}")
print(f"Crank-Nicolson's constant is {out['crank_nicolson_over_the_best']:.2f} times the "
      f"smallest in the sweep")

assert out["order_at_the_cancelling_theta"] > 3.5
assert out["crank_nicolson_is_not_the_most_accurate_at_this_ratio"]
```

**At a fixed $r$, Crank-Nicolson is not the most accurate member of its own family.** Its constant
at $r = 0.4$ is five times the best one available. What Crank-Nicolson has is a property that does
not depend on $r$: its $k$ term vanishes for **every** mesh ratio, so it is second order in time
whatever you choose. The fourth order member is second order in time too, and only fourth order in
space at the one ratio it was tuned for. Lesson 78's $r = 1/6$ result is the same equation read
along the other axis: at $\theta = 0$ the bracket vanishes when $r = 1/6$.

## 3. Two ways to test stability

### 3.1 The matrix method

With Dirichlet ends, one step of the scheme is a matrix,

$$
u^{n+1} = G u^n, \qquad G = (I - \theta r T)^{-1}(I + (1-\theta)r T),
$$

with $T$ the second difference matrix of lesson 75. Stability is the boundedness of $G^n$, and the
two numbers that describe that are the **spectral radius** $\rho(G)$, which governs the limit, and
the **norm** $\lVert G\rVert_2$, which governs the first step.

```python
import numpy as np

print(f"{'theta':>8}{'r':>7}{'rho(G)':>12}{'||G||_2':>12}{'normal':>9}{'stable':>9}")
for theta in (0.0, 0.5, 1.0):
    for r in (0.4, 0.6):
        out = ps.matrix_method(theta, r, 25)
        print(f"{theta:>8.2f}{r:>7.2f}{out['spectral_radius']:>12.6f}"
              f"{out['two_norm']:>12.6f}{str(out['normal']):>9}"
              f"{str(out['stable_in_the_limit']):>9}")
```

Both tests fit in a few lines, and writing them out is the only way to see that they are the
same calculation done in two bases.

```python
def amplification_matrix_from_scratch(theta, r, m):
    """Build G = (I - theta r T)^-1 (I + (1-theta) r T) directly."""
    second = np.zeros((m, m))
    for i in range(m):
        second[i, i] = -2.0
        if i > 0:
            second[i, i - 1] = 1.0
        if i + 1 < m:
            second[i, i + 1] = 1.0
    return np.linalg.solve(np.eye(m) - theta * r * second,
                           np.eye(m) + (1.0 - theta) * r * second)

def von_neumann_from_scratch(theta, r, phi):
    """Substitute u_j^n = g^n exp(i j phi) into the scheme and solve for g."""
    wave = np.exp(1j * np.asarray(phi, dtype=float))
    # the second difference acting on exp(i j phi) multiplies it by this
    symbol = wave - 2.0 + 1.0 / wave
    return ((1.0 + (1.0 - theta) * r * symbol) / (1.0 - theta * r * symbol)).real

for m in (7, 15, 31):
    for theta in (0.0, 0.5, 1.0):
        for r in (0.3, 0.7, 3.0):
            mine = amplification_matrix_from_scratch(theta, r, m)
            theirs = ps.amplification_matrix(theta, r, m)
            assert float(np.max(np.abs(mine - theirs))) < 1e-11
            phases = np.arange(1, m + 1) * np.pi / (m + 1)
            eigen = np.sort(np.linalg.eigvalsh(mine))
            symbols = np.sort(von_neumann_from_scratch(theta, r, phases))
            assert float(np.max(np.abs(eigen - symbols))) < 1e-11
print("the matrix built by hand matches nalib, and its eigenvalues are the symbols")
print("checked at 3 grid sizes, 3 values of theta and 3 mesh ratios")
```

The second assertion is the content of section 3.2 in one line: **the eigenvalues of the matrix
are the von Neumann symbols**, at every size, every $\theta$ and every $r$ tried, to $10^{-11}$.

### 3.2 Von Neumann analysis

Substitute one Fourier mode and the whole scheme collapses to a scalar, which is lesson 78's
$g(\phi)$. It assumes constant coefficients and it ignores the boundaries entirely.

Those look like two different tests. **For the heat equation they are the same statement, exactly.**
$T$ is symmetric, so $G$ is a rational function of a symmetric matrix and is symmetric too; its
eigenvalues are its singular values; and its eigenvectors are exactly the discrete sine modes, so
its eigenvalues are exactly $g(\phi)$ at the phases the grid carries.

```python
out = ps.the_two_tests_agree_on_the_heat_equation()
print(f"{out['cases']} combinations of theta, r and grid size")
print(f"every matrix normal:            {out['every_matrix_is_normal']}")
print(f"spectral radius = 2-norm:       {out['radius_equals_norm_everywhere']}")
print(f"worst eigenvalue disagreement:  {out['worst_eigenvalue_gap']:.3e}")
print(f"both agree with the known limit:{out['and_they_both_say_the_same_about_stability']}")

assert out["the_two_tests_are_the_same_statement"]
assert out["and_they_both_say_the_same_about_stability"]
```

The largest disagreement across 48 combinations is $10^{-14}$, which is rounding. That is why so
much of the literature treats the two tests as interchangeable, and it is worth knowing **why**
they are, so you know when they stop being.

## 4. What non-normality actually costs

Add a convection term, $u_t + a u_x = \alpha u_{xx}$, and difference it centrally. The matrix stops
being symmetric, so $\rho(G)$ and $\lVert G\rVert_2$ part company. The standard warning is that a
non-normal operator with $\rho < 1$ can have powers that **grow before they decay**.

That warning is worth checking rather than repeating. On this scheme it does not happen.

```python
out = ps.the_spectral_radius_is_only_the_limit()
print(f"{'Pe':>6}{'r':>8}{'rho':>10}{'||G||':>10}{'peak ||G^n||':>15}{'normal':>9}"
      f"{'steps to rho':>14}")
for row in out["rows"]:
    steps = "never" if row["steps_to_the_asymptotic_rate"] < 0 \
        else str(row["steps_to_the_asymptotic_rate"])
    print(f"{row['peclet']:>6.2f}{row['r']:>8.4f}{row['spectral_radius']:>10.5f}"
          f"{row['two_norm']:>10.5f}{row['peak_power_norm']:>15.5f}"
          f"{str(row['normal']):>9}{steps:>14}")
print(f"\nnothing grows anywhere in the stable region: {out['nothing_grows_anywhere']}")
print(f"every run is monotone:                      {out['every_run_is_monotone']}")
print(out["note"])

assert out["nothing_grows_anywhere"]
assert out["the_normal_one_hits_its_rate_at_once"]
assert out["the_others_take_longer"]
```

Sweeping the whole stable region, $\lVert G\rVert_2$ comes out just **below** 1 every time, so the
powers decrease monotonically from the first step and there is no transient to find. That is a
negative result and it is reported as one.

What non-normality does cost is the **rate**. The spectral radius is an asymptotic statement,
$\lVert G^n\rVert^{1/n} \to \rho$, and how long that takes is exactly what normality decides:

- pure diffusion, symmetric: the observed rate **is** $\rho$ at $n = 1$;
- $Pe = 1.9$: $\rho = 0.594$, and the observed rate is still $0.9995$ at $n = 20$. It takes
  **244** steps to come within 10 per cent of $\rho$;
- $Pe = 2$ at $r = 0.45$: $\rho = 0.100$, and 400 steps are not enough.

So the spectral radius is right about where the solution ends up and can be badly wrong about how
fast it gets there.

```python
fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.0))
for row in out["rows"]:
    n = np.arange(1, row["power_norms"].size + 1)
    label = f"Pe = {row['peclet']}, r = {row['r']:.3f}"
    left.semilogy(n, row["power_norms"], lw=1.2, label=label)
    left.semilogy(n, row["spectral_radius"] ** n, ":", lw=1.0, color="k", alpha=0.4)
left.set_xlabel("n"); left.set_ylabel("||G^n||"); left.set_ylim(1e-16, 2.0)
left.set_title("powers, against rho^n (dotted)"); left.legend(fontsize=7)

thetas = np.linspace(0.0, 1.0, 201)
for r in (0.2, 0.4, 0.6, 1.0):
    limits = [abs(float(pb.growth_factor(r, np.pi, t))) for t in thetas]
    right.plot(thetas, limits, lw=1.2, label=f"r = {r}")
right.axhline(1.0, color="k", lw=0.8)
right.axvline(0.5, color="k", lw=0.8, ls=":")
right.set_ylim(0.0, 3.0); right.set_xlabel("theta"); right.set_ylabel("|g(pi)|")
right.set_title("the stability boundary of the family"); right.legend(fontsize=8)
fig.tight_layout(); fig.savefig("../figures/79_stability.png", dpi=110); plt.close(fig)
print("saved ../figures/79_stability.png")
```

![Powers of the step operator, and the stability boundary](../figures/79_stability.png)

The left panel is the gap between $\lVert G^n\rVert$ and $\rho^n$: for the symmetric case the two
curves sit on top of each other, and for the others the solid curve hangs above the dotted one for
hundreds of steps. The right panel shows why $\theta = 1/2$ is the threshold: above it the curve
never crosses 1, whatever $r$ is.

## 5. The Lax theorem needs all three cases

Checking the theorem on a scheme that is consistent and stable proves nothing: such a scheme would
converge under any theorem you like. What the statement **forbids** is the other two combinations,
so all three are run.

```python
out = ps.lax_in_three_runs()
print(f"{'case':>28}{'consistent':>12}{'stable':>9}{'converges':>12}   errors")
for case in out["cases"]:
    errors = " ".join(f"{v:.3e}" for v in case["error"])
    print(f"{case['case']:>28}{str(case['consistent']):>12}{str(case['stable']):>9}"
          f"{str(case['converges']):>12}   {errors}")
print(f"\nonly the first converges: {out['only_the_first_converges']}")
print(f"the inconsistent one settles on {out['the_inconsistent_one_settles_on']:.4f}")
print(out["note"])

assert out["only_the_first_converges"]
assert out["the_unstable_one_gets_worse_when_refined"]
```

- **Consistent and stable**: the explicit scheme at $r = 0.4$. The error halves each time the grid
  is refined, at order 2.
- **Consistent and unstable**: the same scheme at $r = 0.6$. The error goes
  $2.7\times10^{3}$, $5.2\times10^{17}$, $7.3\times10^{75}$, then overflow. **Refining makes it
  worse**, which is the exact opposite of convergence.
- **Stable and not consistent**: Du Fort and Frankel at fixed $k/h$. Every run is finite, every
  profile is smooth, and the error settles on $0.0276$ and stays there.

The third is the one to remember. It does not fail to converge; it converges, beautifully, to a
different equation's solution. **Nothing in the run itself gives it away.** Only a comparison
against the exact answer does, and on a real problem there is no exact answer to compare against.
That is why consistency is checked symbolically before the code is written.

## 6. Building your own scheme

Consistency is a set of **linear** conditions on the weights. Expanding

$$
d\,u_{i-1}^{n+1} + e\,u_i^{n+1} + f\,u_{i+1}^{n+1}
= a\,u_{i-1}^{n} + b\,u_i^{n} + c\,u_{i+1}^{n}
$$

about $(x_i, t_n)$ and matching $u$, $u_x$ and $u_{xx}$ in turn gives three conditions: the two
rows sum to the same thing, their first moments agree, and their second moments differ by $2r$.
That last one is where the equation enters. Such conditions can always be met.

**Stability is not linear and cannot be arranged this way.** It has to be checked afterwards, and
it depends on $r$ as well as on the weights.

```python
out = ps.build_your_own_scheme(r=0.25)
print(f"{'scheme':>30}{'2nd moment':>12}{'should be':>11}{'consistent':>12}"
      f"{'order':>8}{'max |g|':>9}{'stable':>8}")
for row in out["rows"]:
    print(f"{row['scheme']:>30}{row['second_moment_condition']:>12.4f}"
          f"{row['second_moment_should_be']:>11.4f}{str(row['consistent']):>12}"
          f"{row['measured_order']:>8.3f}{row['largest_growth']:>9.4f}"
          f"{str(row['stable']):>8}")
print(f"\nconsistency and stability are independent: "
      f"{out['consistency_and_stability_are_independent']}")

assert out["consistency_and_stability_are_independent"]
assert out["the_wrong_second_moment_still_converges_to_something"]
```

Five schemes, and the last two are the point.

- The explicit scheme at $r = 0.75$ satisfies **every** consistency condition, to exactly the same
  order as the others, and has $\lvert g\rvert = 2$ at $\phi = \pi$.
- A scheme whose second moment is $0.75$ instead of $0.5$ fails no stability test at all. It is
  stable, well behaved, and solves $u_t = 1.5\,\alpha u_{xx}$. Its measured order is $-0.000$,
  because the residual does not fall at all.

Neither property is visible in the other's test, which is precisely why Lax's theorem has to
assume consistency rather than derive it.

## 7. Exercises

**Level 1, understanding**

1.1 Define consistency, stability and convergence, each in one sentence, and say which of the
three needs the exact solution.

1.2 State the Lax equivalence theorem and say what it assumes about the problem.

1.3 Explain why the matrix method and von Neumann analysis coincide for the heat equation.

1.4 Say what the spectral radius does and does not tell you about $\lVert G^n\rVert$.

1.5 Explain why a stable inconsistent scheme is more dangerous in practice than an unstable one.

**Level 2, derivation**

2.1 Derive the truncation error of the weighted family and get the bracket
$\left(\tfrac12-\theta\right)r - \tfrac{1}{12}$.

2.2 Derive the three consistency conditions on the six weights of section 6.

2.3 Show that $G$ is symmetric for the weighted family and that its eigenvectors are the discrete
sine modes.

2.4 Derive the two stability conditions $r \le 1/2$ and $rPe^2 \le 2$ for the explicit
convection-diffusion scheme.

2.5 Prove that $\lVert G^n\rVert^{1/n} \to \rho(G)$ for any matrix, and say where normality is
used to make it an equality at $n = 1$.

**Level 3, computational**

3.1 Implement the matrix method for a Neumann boundary and find whether the stability limit
changes.

3.2 Implement a scheme with a variable coefficient $\alpha(x)$ and compare the matrix method
against a frozen coefficient von Neumann analysis at each point.

3.3 Implement the pseudospectrum of $G$ and use it to bound the transient growth, then check the
bound against the measured powers.

3.4 Implement a search over the six weights for the scheme of highest order subject to stability
at a given $r$, and report what it finds.

3.5 Implement a scheme that is consistent, stable and convergent but with an unusably large
constant, and say how you would detect that in practice.

**Level 4, experimental**

4.1 Measure the number of steps to reach the asymptotic decay rate as a function of the Peclet
number, and fit its growth.

4.2 Measure how the stability limit of the explicit scheme changes when the boundary condition is
changed from Dirichlet to Neumann to periodic, using the matrix method for each.

4.3 Measure the constant in the truncation formula for a problem whose solution is not a single
mode, and say what $\max\lvert u_{xxxx}\rvert$ has to be replaced by.

**Level 5, advanced**

5.1 **Where von Neumann fails.** Construct a variable coefficient problem where the frozen
coefficient von Neumann test says stable at every point and the matrix method says unstable, and
explain the mechanism.

5.2 **The Kreiss matrix theorem.** State the equivalence between power boundedness and a resolvent
condition, and use it to explain why the spectral radius is not enough.

5.3 **Lax's theorem is an equivalence.** Prove the easy direction, that an unstable consistent
scheme cannot converge, and identify where the well posedness of the continuous problem is used.

## 8. Key takeaways

- **Measure the constant, not just the order.** The family's truncation error is
  $\alpha h^2\left[\left(\tfrac12-\theta\right)r - \tfrac1{12}\right]u_{xxxx}$, and the measured
  constants match that to three digits at every $\theta$.

- **Crank-Nicolson is not the most accurate member at a fixed $r$.** At $r = 0.4$ its constant is
  five times the smallest available, which belongs to $\theta = 0.2917$. What Crank-Nicolson has
  is second order in time for **every** $r$.

- **For the heat equation the two stability tests are the same statement**, to $10^{-14}$ across
  48 combinations, because $G$ is symmetric and its eigenvectors are the sine modes.

- **Non-normality does not always cause transient growth.** Sweeping the whole stable region of
  the convection-diffusion scheme found none: $\lVert G\rVert_2$ is below 1 every time.

- **What non-normality does cost is the rate.** A symmetric $G$ decays at $\rho$ from step 1. At
  $Pe = 1.9$ the observed rate is still $0.9995$ at step 20 for a matrix with $\rho = 0.594$, and
  it takes 244 steps to get within 10 per cent.

- **The Lax theorem needs its forbidden cases exhibited.** Consistent and unstable gets worse when
  refined, going $10^{3}$, $10^{17}$, $10^{75}$, overflow. Stable and inconsistent settles on an
  error of $0.0276$ and stays there.

- **A stable inconsistent scheme gives nothing away.** It runs, it is smooth, it converges, and it
  converges to the wrong equation. Only the exact solution catches it.

- **Consistency is linear and can always be arranged; stability is not and cannot.** The explicit
  scheme at $r = 0.75$ meets every consistency condition and amplifies by 2 per step.

## Where this goes next

Lesson 80 takes the same schemes into two space dimensions, where the implicit ones stop being
cheap: the matrix is no longer tridiagonal, and a direct solve costs far more than a step should.
The alternating direction implicit method is the answer, and it works by splitting one hard solve
into two easy ones.
