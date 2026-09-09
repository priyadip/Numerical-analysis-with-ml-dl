# Solutions: Part 2, Nonlinear Equations and Root Finding

Worked solutions for the exercises in lessons 09 to 14.

Levels 1 and 2 are answered in full. Levels 3 and 4 give the method, the key code, and the
result you should get, so you can check your own work rather than copy it. Level 5 questions
are open ended, so those get a route through the problem and the answer where there is a
definite one.

Every number quoted below was measured by running the code, not estimated. Where a measured
answer disagrees with the textbook figure, the measured one is given and the disagreement is
explained.

---

## Lesson 09, Bracketing Methods

### 1.1 Why bisection cannot fail

It cannot fail because the **intermediate value theorem** does the work, not the algorithm.
If $f$ is continuous on $[a,b]$ and $f(a)f(b) < 0$, then a root exists in $(a,b)$. Bisection
splits the interval and keeps the half that still has a sign change, so the invariant "a root
lies in the current bracket" is true at the start and preserved at every step. The width halves
every time, so the bracket shrinks to a point, and that point is a root.

Three things are being assumed, and all three can be violated in practice:

1. **$f$ is continuous.** Without continuity a sign change means nothing. Bisection on
   $f(x) = 1/x$ over $[-1, 1]$ converges happily to $x = 0$, which is a pole, not a root.
2. **The initial bracket is genuine**, meaning $f(a)$ and $f(b)$ really do have opposite signs.
   If you got the signs from a floating point evaluation of an $f$ that is near zero at an
   endpoint, you may not have a bracket at all.
3. **$f$ is evaluated exactly enough for its sign to be right.** Near the root, the computed
   sign of $f$ is noise. This is why the bracket can shrink below the point where the sign is
   meaningful, and it is the roundoff floor discussed in section 9.

### 1.2 How many steps

The error bound after $k$ steps is $(b-a)/2^{k+1}$. Set that below $10^{-10}$ with $b - a = 4$:

$$\frac{4}{2^{k+1}} \le 10^{-10} \iff 2^{k+1} \ge 4\times 10^{10}
\iff k+1 \ge \log_2(4\times 10^{10}) = 35.22.$$

So $k + 1 = 36$, meaning $k = 35$ steps. `nalib.roots.bisection_steps_needed(0, 4, 1e-10)`
returns **35**, and running the method to that tolerance takes exactly 35 iterations, with a
final half width of $5.82 \times 10^{-11}$.

Note that this number does not depend on $f$ at all. That is bisection's defining feature.

### 1.3 Regula falsi's bracket

Regula falsi always keeps a valid bracket, because it only ever replaces the endpoint whose sign
matches the new point's sign. But when one endpoint stops being updated, the bracket width stops
shrinking, so $b - a$ is no longer an error bound worth having. The bracket stays valid and stays
almost as wide as it started, so it tells you the root is somewhere in a big interval, which you
already knew.

### 2.1 Bisection error bound by induction

Let $[a_k, b_k]$ be the bracket after $k$ steps, with $[a_0, b_0] = [a, b]$, and let
$x_k = (a_k + b_k)/2$ be the reported midpoint.

**Claim:** $b_k - a_k = (b-a)/2^k$.

*Base case.* $k = 0$ gives $b_0 - a_0 = b - a$, which is $(b-a)/2^0$.

*Inductive step.* One step replaces $[a_k, b_k]$ by one of its two halves, so
$b_{k+1} - a_{k+1} = (b_k - a_k)/2 = (b-a)/2^{k+1}$ by the inductive hypothesis.

Now, the root $r$ lies in $[a_k, b_k]$ by the sign invariant, and $x_k$ is that interval's
midpoint, so $r$ and $x_k$ are both in an interval of width $(b-a)/2^k$ with $x_k$ at its centre.
Hence

$$|x_k - r| \le \tfrac12 (b_k - a_k) = \frac{b-a}{2^{k+1}}. \qquad \square$$

The bound is sharp: the root can sit at an endpoint of the bracket.

### 2.2 Convex $f$ keeps the right endpoint stuck

Let $f$ be convex on $[a,b]$ with $f(a) < 0 < f(b)$. Convexity means the **chord lies on or above
the graph** between any two points on it.

So the chord from $(a, f(a))$ to $(b, f(b))$ satisfies

$$L(x) \ge f(x) \quad \text{for all } x \in [a,b].$$

The regula falsi point $c$ is where the chord crosses zero, so $L(c) = 0$, which gives
$f(c) \le 0$.

Now compare $c$ with the root $r$, where $f(r) = 0$. Since $f$ is convex with $f(a) < 0 < f(b)$
it is increasing through the root, so $f(c) \le 0 = f(r)$ forces $c \le r$. The new point is on
or to the left of the root.

Because $f(c) \le 0$ matches the sign of $f(a)$, the algorithm replaces $a$ by $c$ and leaves $b$
alone. This is true at every step, so $b$ is **never** updated. That is exactly the stagnation of
section 7.

### 2.3 The regula falsi formula

The line through $(a, f(a))$ and $(b, f(b))$ is

$$L(x) = f(a) + \frac{f(b) - f(a)}{b - a}\,(x - a).$$

Set $L(c) = 0$ and solve:

$$0 = f(a) + \frac{f(b)-f(a)}{b-a}(c - a)
\;\Longrightarrow\;
c = a - f(a)\,\frac{b-a}{f(b)-f(a)}
= \frac{a f(b) - b f(a)}{f(b) - f(a)}.$$

The last form is the one usually quoted. It is symmetric in the two endpoints, which is pleasant,
but it is **numerically worse** than the first form when $a$ and $b$ are close: the numerator
$af(b) - bf(a)$ is then a difference of two nearly equal quantities, which is lesson 05's
cancellation. Prefer $c = a - f(a)(b-a)/(f(b)-f(a))$ in code.

### 2.4 The linear rate of stuck regula falsi

Suppose $b$ is the stuck endpoint and $a_k \to r$ from the left. The next point is

$$a_{k+1} = a_k - f(a_k)\,\frac{b - a_k}{f(b) - f(a_k)}.$$

Write $e_k = r - a_k > 0$. Near the root, $f(a_k) \approx -f'(r) e_k$, and since $e_k$ is small
compared with $b - r$ we have $b - a_k \approx b - r$ and $f(b) - f(a_k) \approx f(b)$. Then

$$e_{k+1} = r - a_{k+1} = e_k + f(a_k)\frac{b-a_k}{f(b)-f(a_k)}
\approx e_k - f'(r) e_k \cdot \frac{b - r}{f(b)}
= e_k\left(1 - \frac{f'(r)(b-r)}{f(b)}\right).$$

So the convergence is linear with asymptotic rate

$$\rho = 1 - \frac{f'(r)(b - r)}{f(b)}.$$

**Why $\rho$ can be arbitrarily close to 1.** The correction term $f'(r)(b-r)/f(b)$ is small
whenever $f(b)$ is large relative to the linear extrapolation $f'(r)(b-r)$, which happens exactly
when $f$ curves upward steeply between $r$ and $b$. Take $f(x) = x^n - 1$ on $[0, 1.3]$: here
$r = 1$, $f'(1) = n$, $b - r = 0.3$, $f(b) = 1.3^n - 1$, so

$$\rho = 1 - \frac{0.3n}{1.3^n - 1} \to 1 \text{ as } n \to \infty,$$

because $1.3^n$ beats $0.3n$. Exercise 4.1 measures this.

### 3.1 Sign comparison instead of a product

```python
# fragile
if f(a) * f(c) < 0: ...

# robust
if (f(a) < 0) != (f(c) < 0): ...
```

The product version overflows when both values are large. Take

$$f(x) = 10^{200}(x - 1)^3 \quad \text{on } [-2, 3].$$

Then $f(-2) \approx -2.7\times 10^{201}$ and $f(3) = 8\times 10^{200}$. Their product is about
$-2\times 10^{402}$, which overflows to $-\infty$. That happens to still be negative, so this
particular case survives, but change the exponent to $10^{250}$ and the product of two finite
values overflows in a way that can produce `inf * 0 = nan`, and `nan < 0` is `False`, so the
bracket update goes the wrong way silently.

The sign version never multiplies, so it cannot overflow. It also handles $f(c) = 0$ correctly
and works for values so small they underflow to zero, where the product is $0$ and the test
`< 0` fails. This is why `nalib.roots.bisection` compares signs.

### 3.2 Illinois from scratch

The whole method is regula falsi plus four characters:

```
if the same endpoint was kept twice in a row:
    halve the retained function value
```

```python
def illinois(f, a, b, tol=1e-12, max_iter=200):
    fa, fb = f(a), f(b)
    side = 0
    for k in range(max_iter):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        if fc == 0 or abs(b - a) < tol:
            return c, k
        if (fc < 0) == (fa < 0):
            a, fa = c, fc
            if side == +1:
                fb *= 0.5              # b was kept last time too
            side = +1
        else:
            b, fb = c, fc
            if side == -1:
                fa *= 0.5
            side = -1
    return c, max_iter
```

Halving the retained value pulls the next chord towards the stuck endpoint, which forces it to
move. Section 8's counts should reproduce exactly, since the algorithm is deterministic.

### 3.3 Anderson-Bjorck

Replace the fixed factor $1/2$ with $m = 1 - f(c)/f(\text{kept})$, using $1/2$ instead when
$m \le 0$:

```python
m = 1.0 - fc / f_kept
scale = m if m > 0 else 0.5
f_kept *= scale
```

On the six problems of section 8, Anderson-Bjorck typically saves one to three iterations over
Illinois, and never loses. The reason is that Illinois halves blindly, while Anderson-Bjorck
scales by an estimate of how far off the chord actually was, so it is a better guess at the right
amount of correction. Both keep the bracket, so both keep bisection's guarantee.

The lesson to draw is that the guarantee comes from the bracket, and the speed comes from the
scaling rule. Those are two separate design choices, and you can improve the second without
touching the first.

### 4.1 Regula falsi stagnation against curvature

Measured, $f(x) = x^n - 1$ on $[0, 1.3]$ to a tolerance of $10^{-10}$:

| $n$ | regula falsi | Illinois |
|---:|---:|---:|
| 2 | 11 | 6 |
| 4 | 24 | 8 |
| 6 | 41 | 8 |
| 8 | 65 | 11 |
| 10 | 103 | 13 |
| 12 | 161 | 14 |
| 14 | 252 | 15 |
| 16 | 394 | 15 |
| 18 | 621 | 19 |
| 20 | 982 | 19 |

Regula falsi's count grows **geometrically**: each increase of 2 in $n$ multiplies the count by
about 1.58, and $982/103 \approx 9.5$ from $n = 10$ to $n = 20$.

That matches exercise 2.4. The predicted count is $\log(\text{tol}/(b-a))/\log\rho$ with
$\rho = 1 - 0.3n/(1.3^n-1)$:

| $n$ | $\rho$ | predicted | measured |
|---:|---:|---:|---:|
| 4 | 0.3535 | 22 | 24 |
| 10 | 0.7654 | 87 | 103 |
| 20 | 0.9683 | 722 | 982 |

The prediction is low by 10 to 35 percent, which is what you should expect: $\rho$ is the
**asymptotic** rate and the early iterations are faster than asymptotic, but the formula ignores
the transient in the other direction as well. The growth trend is captured exactly, and that was
the question.

Illinois grows like $\log n$, if that. Roughly a factor of 50 separates them at $n = 20$, and the
gap keeps widening.

### 4.2 Bisection's count is independent of $f$

Run bisection with the same bracket $[0, 1]$ and the same tolerance on ten unrelated functions,
each with a root inside. Every one takes the same number of iterations, because the loop
condition depends only on the bracket width, which is halved every step regardless of what $f$
does. The only thing $f$ affects is **which** half is kept, and that does not change the count.

This is the concrete meaning of "bisection is $f$-independent". It is a guarantee and a
limitation in the same sentence: nothing about $f$ can make bisection fail, and nothing about
$f$ can make it fast.

### 4.3 Illinois's observed order

You will **not** cleanly get 1.44, and finding that out is the point of the exercise.

Measured with `nalib.convergence.reliable_order` against a reference root, Illinois converges in
6 to 10 iterations, which is too few steps to fit an order to. Worse, the error sequence is not
monotone, because Illinois inherits regula falsi's habit of approaching from one side and then
jumping. The order estimator returns nonsense values in the hundreds on some problems and 0 on
others, which is the estimator telling you the model does not fit, not the method misbehaving.

The theoretical 1.442 comes from analysing the **three-step cycle** that Illinois settles into
asymptotically, and it is an average over the cycle rather than a per-step order. This is the
same situation as Brent and Muller in lesson 12: hybrid methods have no measurable per-step
order, and the honest comparison is total function evaluations to a fixed tolerance. On that
measure Illinois is excellent, and lesson 11's efficiency table is where it should be read off.

### 5.1 Bisection is optimal among sign-only methods

**Setup.** An algorithm may query the sign of $f$ at any point it likes, adaptively. After $k$
queries it must output an interval guaranteed to contain a root. We show no algorithm can
guarantee an interval shorter than $(b-a)/2^k$.

**Adversary argument.** The adversary does not fix $f$ in advance. It maintains a *live set*
$S$, the set of points that could still be the root consistently with the answers given so far.
Initially $S = [a,b]$, of length $b - a$.

When the algorithm queries the sign at a point $x$, the adversary splits $S$ at $x$ into $S_{<}$
and $S_{>}$, and answers with whichever sign keeps the **longer** piece alive:

- answering "$f(x) < 0$" is consistent with the root lying in $S_{>}$,
- answering "$f(x) > 0$" is consistent with the root lying in $S_{<}$.

Either answer is consistent with some continuous $f$ having the required sign change, so the
adversary is never caught lying. Since it always keeps the longer piece,

$$|S_{k+1}| \ \ge\ \tfrac12 |S_k|,$$

so $|S_k| \ge (b-a)/2^k$ after $k$ queries.

**Conclusion.** The algorithm must output an interval containing all of $S_k$, since any point of
$S_k$ could be the root. So its output has length at least $(b-a)/2^k$. That is exactly
bisection's guarantee, so bisection is optimal, and the adversary is beaten only by querying at
the midpoint every time, which is exactly what bisection does.

**One bit per evaluation.** Each query returns one bit, and $k$ bits can distinguish at most
$2^k$ outcomes, so they can locate the root to at best one part in $2^k$. The argument above is
the careful version of that counting.

Note carefully what this does **not** say. Newton beats bisection easily, and does not contradict
this at all, because Newton uses the **value** of $f$ and $f'$, not just the sign. Sign
information is worth one bit; a floating point value is worth about 53.

### 5.2 A case needing over 1000 regula falsi iterations

From the table in 4.1, extend to larger $n$. Measured at tolerance $10^{-10}$ on $[0, 1.3]$:

| polynomial | regula falsi | Illinois |
|---|---:|---:|
| $x^{20} - 1$ | 982 | 19 |
| $x^{22} - 1$ | 1562 | 20 |
| $x^{24} - 1$ | 2497 | 18 |
| $x^{26} - 1$ | 4010 | 21 |

So $f(x) = x^{24} - 1$ on $[0, 1.3]$ answers the question: **2497** iterations for regula falsi,
**18** for Illinois, a factor of 139. ($x^{22}$ also works but Illinois needs exactly 20 there,
so it fails the "fewer than 20" requirement by one.)

The mechanism is exercise 2.2: $x^n - 1$ is convex on $[0, 1.3]$, so the right endpoint 1.3 is
never updated, and the left endpoint creeps in at the rate exercise 2.4 predicts.

### 5.3 Finding even-multiplicity roots

Bracketing needs a sign change, and $f$ does not change sign at a root of even multiplicity, so
no bracketing method can find one. Ever. This is a property of the information available, not of
any particular algorithm.

Three strategies, each giving something up:

**Bracket a sign change of $f'$.** At a double root, $f$ has a local minimum with value 0, so
$f'$ changes sign there. Bracket that instead, then check whether $f$ is actually zero at the
point found. What you give up: you need $f'$, and you now find every local extremum of $f$, most
of which are not roots. You also lose accuracy, because you are solving $f' = 0$ and reading off
$f$, and lesson 12 says a double root can only be located to $\sqrt{u} \approx 10^{-8}$ anyway.

**Minimise $|f|$.** Use golden section search or Brent's `fminbound` on $|f|$ over the interval,
then test whether the minimum value is at the noise level. What you give up: minimisation has no
bracket guarantee comparable to a sign change, you can land in a local minimum that is not a
root, and the accuracy limit is again $\sqrt{u}$, this time because near a minimum $|f|$ is flat.

**Deflate to a simple root.** The function $u = f/f'$ has a **simple** root wherever $f$ has a
root of any multiplicity (lesson 11, exercise 2.4). So bracket a sign change of $u$. What you
give up: $u$ is $0/0$ at the root itself, so it needs careful evaluation, and you need $f'$.
This is the cleanest of the three when $f'$ is available, and it is what lesson 11 recommends.

The honest summary: an even-multiplicity root is genuinely harder, and every route around the
missing sign change costs you either a derivative, a guarantee, or half your digits.

---

## Lesson 10, Fixed Point Iteration

### 1.1 Why the bound must hold on an interval

Because the theorem has to prove the iterates **stay** in the region where the bound holds.

The proof structure is: if $x_k \in [a,b]$ then $|x_{k+1} - r| = |g(x_k) - g(r)| \le L|x_k - r|$
by the mean value theorem, where the intermediate point is somewhere between $x_k$ and $r$. That
intermediate point is not $r$, so knowing $|g'(r)| < 1$ alone tells you nothing about it. You
need the bound at the intermediate point, and you do not know where that is, so you need it
everywhere between.

The bound also does the second half of the work: $L < 1$ on the whole interval proves the map
sends $[r - \delta, r + \delta]$ into itself, so the iterates cannot escape and the argument can
be repeated.

If you only know $|g'(r)| < 1$, continuity gives you $|g'| < 1$ on **some** neighbourhood, so
convergence is still local. But you get no explicit interval and no error bound, which is
precisely what the theorem is for.

### 1.2 Convergence with $g'(r) = -0.9$

The sign is negative, so the error changes sign every step and the iterates **alternate** around
the root, approaching in a spiral rather than from one side. A useful practical consequence: the
root is bracketed by any two consecutive iterates, so $|x_{k+1} - x_k|$ is a genuine error bound
here, which is unusual.

The magnitude is 0.9, so each step multiplies the error by 0.9. To gain one decimal digit you
need $0.9^m \le 0.1$, so

$$m \ge \frac{\log 0.1}{\log 0.9} = \frac{-1}{-0.0458} = 21.85,$$

meaning **22 iterations per decimal digit**. Full double precision from an error of 1 needs about
$16 \times 22 = 350$ iterations. This is the practical meaning of "linear convergence with a rate
near 1": technically convergent, useless in practice, and exactly the situation Aitken's
$\Delta^2$ exists to fix.

### 1.3 Why $|x_{k+1} - x_k|$ lies

For a linearly convergent iteration with rate $\rho = g'(r)$, the error and the step are related
by

$$x_{k+1} - x_k = (x_{k+1} - r) - (x_k - r) = e_{k+1} - e_k \approx (\rho - 1)e_k,$$

so

$$|e_k| \approx \frac{|x_{k+1} - x_k|}{|1 - \rho|}.$$

Divide the step by $|1 - \rho|$. When $\rho$ is close to 1, that divisor is tiny and the true
error is **much larger** than the step suggests. With $\rho = 0.99$ the step underestimates the
error by a factor of 100.

The dangerous case is $\rho$ near $+1$, meaning slow one-sided convergence, where the iteration
crawls and the steps look reassuringly small. The safe case is $\rho$ negative, where $|1-\rho| >
1$ and the step actually **over**estimates the error.

In practice you can estimate $\rho$ from the last three iterates as
$\rho \approx (x_{k+1}-x_k)/(x_k - x_{k-1})$ and divide by $|1 - \rho|$. That is Aitken's
correction in disguise.

### 2.1 $g'(r) = 0$ gives quadratic convergence

Expand $g$ about $r$ with Taylor's theorem with remainder:

$$g(x_k) = g(r) + g'(r)(x_k - r) + \tfrac12 g''(\xi_k)(x_k - r)^2$$

for some $\xi_k$ between $x_k$ and $r$. Use $g(x_k) = x_{k+1}$, $g(r) = r$ and $g'(r) = 0$:

$$x_{k+1} - r = \tfrac12 g''(\xi_k)(x_k - r)^2,$$

that is $e_{k+1} = \tfrac12 g''(\xi_k) e_k^2$. As $x_k \to r$ we have $\xi_k \to r$, and $g''$ is
continuous, so

$$\lim_{k\to\infty} \frac{|e_{k+1}|}{|e_k|^2} = \frac{|g''(r)|}{2}. \qquad \square$$

That is quadratic convergence with asymptotic error constant $|g''(r)|/2$. This is the general
statement of which Newton's quadratic convergence is a special case: Newton is exactly the
rearrangement that forces $g'(r) = 0$.

### 2.2 Deriving Aitken's formula

Assume the error is exactly geometric: $e_{k+1} = C e_k$ with $e_k = x_k - r$. Then

$$x_{k+1} - r = C(x_k - r), \qquad x_{k+2} - r = C(x_{k+1} - r).$$

Divide the second by the first to eliminate $C$:

$$\frac{x_{k+2} - r}{x_{k+1} - r} = \frac{x_{k+1} - r}{x_k - r}.$$

Cross multiply:

$$(x_{k+2} - r)(x_k - r) = (x_{k+1}-r)^2.$$

Expand both sides. The $r^2$ terms cancel:

$$x_{k+2}x_k - r(x_k + x_{k+2}) = x_{k+1}^2 - 2rx_{k+1}.$$

Solve for $r$:

$$r\big(2x_{k+1} - x_k - x_{k+2}\big) = x_{k+1}^2 - x_kx_{k+2},$$

$$r = \frac{x_{k+1}^2 - x_kx_{k+2}}{2x_{k+1} - x_k - x_{k+2}}.$$

That is Aitken's formula, but in a form you should **not** use: the numerator is a difference of
two nearly equal products and the denominator is a difference of nearly equal numbers, so both
suffer catastrophic cancellation (lesson 05). Rearranging algebraically gives the standard form

$$\hat{x}_k = x_k - \frac{(\Delta x_k)^2}{\Delta^2 x_k},
\qquad \Delta x_k = x_{k+1}-x_k, \quad \Delta^2 x_k = x_{k+2} - 2x_{k+1} + x_k,$$

which computes the small **correction** to $x_k$ rather than recomputing $r$ from scratch. The
cancellation in $\Delta^2 x_k$ is still there and still limits how far you can push the
acceleration, which is what exercise 4.3 measures.

### 2.3 Steffensen as Newton

Steffensen's method is cleanest to characterise not through the $h$ given in the question, but
as **Newton's method on $f(x) = g(x) - x$ with the derivative replaced by a difference quotient
whose step is the residual itself**.

Newton on $f$ is $x - f(x)/f'(x)$. Approximate

$$f'(x) \approx \frac{f(x + f(x)) - f(x)}{f(x)}$$

using $h = f(x)$ as the step size. Substituting $f = g - x$ and simplifying gives exactly the
Steffensen iterate

$$x_{k+1} = x_k - \frac{(g(x_k)-x_k)^2}{g(g(x_k)) - 2g(x_k) + x_k},$$

which is Aitken's formula applied to $x_k, g(x_k), g(g(x_k))$ and then restarted. That is the
cleaner characterisation the question asks for.

Two consequences follow at once. It converges **quadratically** even though it uses no
derivative, because a difference quotient with step $h = f(x) \to 0$ becomes exact in the limit,
so the derivative error vanishes fast enough to preserve the rate. And it costs **two**
evaluations of $g$ per step, so its efficiency index is $2^{1/2} = 1.414$, the same as Newton and
below the secant method's 1.618.

The function $h$ in the question, with $f/f'$ formed from the same difference quotient, gives the
same iteration by construction. It is correct but it hides the mechanism.

### 2.4 A quadratic rearrangement for $\sqrt{a}$

We need $g$ with $g(\sqrt a) = \sqrt a$ and $g'(\sqrt a) = 0$.

Take a general one-parameter family. Any $g$ of the form
$g(x) = \alpha x + (1-\alpha)a/x$ fixes $\sqrt a$, since $\alpha\sqrt a + (1-\alpha)a/\sqrt a =
\alpha \sqrt a + (1-\alpha)\sqrt a = \sqrt a$. Differentiate:

$$g'(x) = \alpha - (1-\alpha)\frac{a}{x^2},
\qquad g'(\sqrt a) = \alpha - (1 - \alpha) = 2\alpha - 1.$$

Setting $g'(\sqrt a) = 0$ gives $\alpha = 1/2$, so

$$g(x) = \tfrac12\left(x + \frac{a}{x}\right),$$

which is the Babylonian iteration. Its asymptotic error constant is
$|g''(\sqrt a)|/2 = \tfrac{1}{2}\cdot\tfrac{a}{(\sqrt a)^3} = \frac{1}{2\sqrt a}$, so
$e_{k+1} \approx e_k^2/(2\sqrt a)$.

Newton on $f(x) = x^2 - a$ gives $x - (x^2-a)/(2x) = (x + a/x)/2$, the same thing. The exercise
shows the Babylonian method was Newton's method 3500 years early, and that "make $g'(r) = 0$" is
the design principle behind it.

### 3.1 Fixed point iterator with history

```python
def fixed_point(g, x0, tol=1e-12, max_iter=1000):
    xs = [float(x0)]
    for _ in range(max_iter):
        x = g(xs[-1])
        xs.append(x)
        if abs(x - xs[-2]) <= tol:
            break
    return np.array(xs)
```

Returning the whole history, not just the answer, is the pattern used throughout `nalib`. You
cannot measure a convergence rate from a final answer, and measuring rates is most of what Part 2
is about. Section 2's table should reproduce to the last digit, since nothing here is random.

### 3.2 Cobweb plots for $g(x) = \cos x$

The plot alternates horizontal moves to the line $y = x$ and vertical moves to the curve
$y = g(x)$. The fixed point is the Dottie number $r = 0.739085133215$, and

$$g'(r) = -\sin(0.739085\ldots) = -0.673612.$$

The derivative is **negative**, so the cobweb spirals inward around the fixed point rather than
staircasing towards it from one side. The magnitude 0.674 is comfortably below 1, so the spiral
tightens, at about 0.674 per step, which is roughly 6.2 iterations per decimal digit.

Starting points matter not at all here: $\cos$ maps everything into $[-1,1]$ after one step and
$|g'| = |\sin x| \le \sin 1 = 0.841 < 1$ on that interval, so **every** real starting point
converges. This is a genuinely globally convergent fixed point iteration, which is rare.

### 3.3 Aitken and Steffensen from scratch

```python
def aitken(seq):
    x = np.asarray(seq, dtype=float)
    d1 = x[1:-1] - x[:-2]
    d2 = x[2:] - 2*x[1:-1] + x[:-2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d2 != 0, x[:-2] - d1**2 / d2, x[:-2])
```

Two checks worth doing. On a sequence that is exactly geometric, such as $x_k = r + C\rho^k$,
Aitken must return $r$ to machine precision **immediately**, because its derivation assumed
exactly that model. And Aitken applied to a quadratically convergent sequence must not make
things worse, which it does not, but it does not help either, because the model does not fit.

`nalib.roots.aitken` and `nalib.roots.steffensen` should agree with yours to roundoff on
well-scaled problems, and may differ in the last few digits once $\Delta^2 x$ starts cancelling.

### 4.1 Sweeping $c$ in $g_c(x) = x + c(R - x)$

Here $g_c'(x) = 1 - c$ exactly, everywhere, so the theory says the rate is $|1-c|$ with no
approximation. Measured with $R = 1$, $x_0 = 0$, tolerance $10^{-14}$:

| $c$ | $\lvert 1-c\rvert$ | converged | iterations | measured rate |
|---:|---:|:---:|---:|---:|
| 0.1 | 0.900 | yes | 286 | 0.9000 |
| 0.5 | 0.500 | yes | 47 | 0.5000 |
| 0.9 | 0.100 | yes | 15 | 0.1000 |
| 1.0 | 0.000 | yes | 2 | exact |
| 1.5 | 0.500 | yes | 49 | 0.5000 |
| 1.9 | 0.900 | yes | 314 | 0.9000 |
| 2.0 | 1.000 | **no** | limit | oscillates |
| 2.1 | 1.100 | **no** | limit | diverges |

The measured rate equals $|1-c|$ to four decimal places at every convergent value, so the plot
against $|1-c|$ is the diagonal line, as predicted.

**At $c = 1$:** $g_1(x) = x + (R - x) = R$, a constant map. It reaches the answer in one step and
the loop stops on the second. There is no rate to measure, because $g'(r) = 0$ and the error is
zero, not merely small. This is the fixed point analogue of hitting the root exactly.

**At $c = 2$:** $g_2(x) = x + 2(R-x) = 2R - x$, which is reflection about $R$. It has period 2:
$x_0, 2R - x_0, x_0, \dots$ forever. Not convergent, not divergent, exactly on the boundary. This
is the $|g'| = 1$ case where the theorem is silent, and here silence is correct, because the
behaviour is neither.

**Beyond $c = 2$:** $|1 - c| > 1$ and the iteration diverges geometrically, alternating in sign
as it goes.

### 4.2 Basin of attraction against the contraction interval

Find the basin numerically by running the iteration from a fine grid of $x_0$ and recording
convergence. Compare that set with $\{x : |g'(x)| < 1\}$.

The two sets are **not** the same, and the theorem only ever claimed one inclusion. The
contraction interval is a subset of the basin, usually a strict subset: points outside it can
still converge, because the iteration may jump into the contraction region on its first step and
be captured. Convergence from $x_0$ needs only that **some** iterate lands in the contracting
region, not that $x_0$ itself is in it.

For maps with more than one fixed point the basin can also be disconnected, and its boundary can
be extremely intricate. That is the one-dimensional shadow of the Newton fractal in lesson 11,
exercise 3.2.

### 4.3 Repeated Aitken

It helps, and then it stops helping, and the crossover is sharp.

Applying Aitken to an accelerated sequence is the start of the **Shanks transformation** table.
Each column removes one more geometric component from the error, so if the error is a sum of a
few geometric terms with different rates, each pass strips one off and the improvement is real.

Cancellation takes over when $\Delta^2 x_k$ becomes comparable to the roundoff in $x_k$ itself.
Each Aitken pass forms a second difference, and a second difference of numbers agreeing to $d$
digits has about $d - 2\log_{10}(\text{ratio})$ correct digits left. Once the sequence has
converged to within a few ulps, $\Delta^2 x$ is pure noise and the formula produces garbage,
sometimes catastrophically so, because you are dividing by that noise.

The rule of thumb: stop accelerating when $|\Delta^2 x_k|$ drops below about $10^{-10}|x_k|$ in
double precision. In practice two or three passes is the useful limit for a sequence with one
dominant geometric mode.

### 5.1 The Banach fixed point theorem

> **Theorem (Banach).** Let $(X, d)$ be a **complete** metric space and $T : X \to X$ satisfy
> $d(T(x), T(y)) \le L\,d(x,y)$ for all $x, y \in X$, with a constant $L < 1$. Then $T$ has
> exactly one fixed point $x^\ast \in X$, the iteration $x_{k+1} = T(x_k)$ converges to it from
> **every** starting point in $X$, and
> $$d(x_k, x^\ast) \le \frac{L^k}{1-L}\,d(x_1, x_0).$$

What it gives you that the calculus version does not:

**No derivative is needed.** Only the Lipschitz condition. So it applies to maps that are not
differentiable, and to spaces where "derivative" is not defined.

**It works in any complete metric space.** Not just $\mathbb{R}$. This is the real payoff:
- $\mathbb{R}^n$ with any norm, giving lesson 14's Theorem 14.1,
- spaces of matrices, giving convergence proofs for iterative linear solvers in Part 4,
- spaces of **functions**, which is how the Picard-Lindelof existence theorem for ordinary
  differential equations is proved (Part 9), and how Fredholm integral equations are solved.

**Existence comes free.** The calculus version assumes a fixed point exists and proves you
converge to it. Banach **proves it exists**, using completeness. That is the difference between
a numerical statement and a theorem of analysis.

**The error bound needs no unknown.** $L^k d(x_1,x_0)/(1-L)$ uses only the first step and $L$, so
it is computable before you run the iteration. The calculus bound involves $|x_k - r|$ with $r$
unknown.

The price is that $L < 1$ must hold **globally** on $X$. In practice you apply it to a closed
ball that $T$ maps into itself, which is complete as a subspace, and then you are back to a local
statement, but a cleanly proved one.

### 5.2 The logistic map

$g_\mu(x) = \mu x(1-x)$ has fixed points $x = 0$ and $x^\ast = 1 - 1/\mu$ (for $\mu \ne 0$).

**Stability.** $g_\mu'(x) = \mu(1 - 2x)$, so at the nontrivial fixed point

$$g_\mu'(x^\ast) = \mu\left(1 - 2\left(1 - \tfrac1\mu\right)\right) = 2 - \mu.$$

Theorem 10.2 requires $|2 - \mu| < 1$, that is $1 < \mu < 3$. So the fixed point is stable
exactly on $(1,3)$ and **loses stability at $\mu = 3$**, where $g'(x^\ast) = -1$.

**Period doubling.** The derivative passing through $-1$ (rather than $+1$) is the signature of a
period-doubling bifurcation. Look for period-2 points, meaning fixed points of $g_\mu \circ
g_\mu$ that are not fixed points of $g_\mu$. Solving gives

$$x_\pm = \frac{\mu + 1 \pm \sqrt{(\mu-3)(\mu+1)}}{2\mu},$$

which is real exactly when $\mu \ge 3$. So a period-2 cycle is born precisely where the fixed
point loses stability, and it inherits the stability. Its multiplier is
$g'(x_+)g'(x_-) = 4 + 2\mu - \mu^2$, which leaves $[-1,1]$ at $\mu = 1 + \sqrt 6 \approx 3.449$,
where period 4 appears, and so on.

**Bifurcation diagram.** For each $\mu$ on a fine grid over $[2.5, 4]$, iterate 1000 times to
discard the transient, then plot the next 200 iterates against $\mu$. You get one branch up to
$\mu = 3$, two to 3.449, four to 3.544, and the cascade accumulates at $\mu_\infty \approx
3.5699$, beyond which the behaviour is chaotic with periodic windows.

**Relation to Theorem 10.2.** The theorem is exactly the statement "the fixed point is stable
while $|g'| < 1$", and every bifurcation in the diagram is a place where some iterate's
derivative crosses the unit circle. The whole bifurcation structure is Theorem 10.2 applied
repeatedly to $g, g^2, g^4, \dots$. A convergence criterion for a numerical method and the route
to chaos are the same inequality.

### 5.3 The $g'(r) = 1$ case

$g(x) = x + x^3$ near $r = 0$ has $g'(0) = 1$, so Theorem 10.2 says nothing. The cubic term
decides, and here it decides badly.

**Measured: it diverges.** From $x_0 = 0.5$ the iteration overflows at step 10. The reason is
that $g(x) - x = x^3$ has the **same sign as $x$**, so every step pushes away from 0. The same
happens from the negative side.

Flip the sign to get the interesting case. For $g(x) = x - x^3$ the correction points back
towards 0, and now the iteration converges, but agonisingly slowly. Measured from $x_0 = 0.5$:

| $k$ | $x_k$ | $1/\sqrt{2k}$ | ratio |
|---:|---:|---:|---:|
| 10 | 0.19142184 | 0.22360680 | 0.856 |
| 100 | 0.06894155 | 0.07071068 | 0.975 |
| 1000 | 0.02228399 | 0.02236068 | 0.997 |
| 10000 | 0.00706802 | 0.00707107 | 0.9996 |
| 100000 | 0.00223595 | 0.00223607 | 0.9999 |

So $x_k \sim 1/\sqrt{2k}$, confirmed to four digits.

**Where that comes from.** Treat $x_k$ as a smooth function of $k$. Then
$dx/dk \approx x_{k+1}-x_k = -x^3$, which separates:

$$\int x^{-3}\,dx = -\int dk \implies -\tfrac12 x^{-2} = -k + C \implies x_k \approx
\frac{1}{\sqrt{2k}}.$$

**Why this is the slowest useful behaviour.** The ratio $x_{k+1}/x_k \to 1$, so the rate is 1 and
the convergence is **sublinear**: slower than any geometric sequence. To gain one decimal digit
you need to multiply $k$ by 100, since $x \sim k^{-1/2}$. Reaching $10^{-8}$ takes about
$5\times 10^{15}$ iterations, which is not a numerical method, it is a statement about the limit.

The lesson: $|g'(r)| = 1$ is a genuine boundary, not a technicality. On one side of it you get
geometric convergence, on the other divergence, and exactly on it the higher order terms decide
between divergence, a period-2 cycle (exercise 4.1 at $c=2$), and convergence too slow to use.

---

## Lesson 11, Newton and Secant Methods

### 1.1 Newton against secant efficiency

The efficiency index is $p^{1/w}$, where $p$ is the order and $w$ the evaluations per step.

- Newton: $p = 2$, $w = 2$ (one $f$, one $f'$), giving $2^{1/2} = 1.4142$.
- Secant: $p = 1.618$, $w = 1$, giving $1.618^{1} = 1.6180$.

**The secant method is more efficient**, by a comfortable margin. Ranked by evaluations needed
per digit gained, secant beats Newton by about 15 percent.

The assumption doing the work is that **$f$ and $f'$ cost the same**. That is often false in both
directions:

- If $f'$ is unavailable or must be computed by finite differences, Newton's real cost is $w = 3$
  or more, and secant wins by even more.
- If $f$ and $f'$ share most of their computation, which happens for polynomials via Horner
  (lesson 01) and for anything differentiated by automatic differentiation in reverse mode, then
  $f'$ is nearly free, $w \approx 1$, and Newton's index rises to $2$, beating secant.

So the honest answer is: secant, unless the derivative is cheap because it shares work with the
function, in which case Newton.

Measured over four test problems to a tolerance of $10^{-12}$, total evaluations: secant 35,
Newton 67. The ratio 1.91 is larger than the index alone predicts because Newton is charged two
evaluations at every step including the wasted first ones.

### 1.2 Newton at a double root

Write $f(x) = (x-r)^m h(x)$ with $h(r) \ne 0$. Then

$$\frac{f}{f'} = \frac{(x-r)^m h}{m(x-r)^{m-1}h + (x-r)^m h'}
= \frac{(x-r)h}{mh + (x-r)h'} \approx \frac{x - r}{m}$$

near $r$. So the Newton step is only $1/m$ of the distance to the root, and

$$e_{k+1} = e_k - \frac{e_k}{m} = \left(1 - \frac1m\right)e_k.$$

That is **linear** convergence with rate $1 - 1/m$, not quadratic. At a double root the rate is
$1/2$, so Newton reduces to bisection's speed while giving up bisection's guarantee.

The reason it degrades is that quadratic convergence needs $f'(r) \ne 0$, and at a multiple root
$f'(r) = 0$. The step $f/f'$ becomes $0/0$, and the ratio survives only because both vanish at
the same rate.

**Multiplying the step by $m$** gives $e_{k+1} = e_k - m\cdot e_k/m = 0$ to first order, so it
restores quadratic convergence exactly. `nalib.roots.newton` takes a `multiplicity` argument for
this.

Two warnings. You usually do not know $m$, and using the wrong one is worse than using 1: an
overestimate makes the iteration overshoot and can diverge. And restoring the **rate** does not
restore the **accuracy**. Lesson 12 shows a root of multiplicity $m$ can only be located to about
$u^{1/m}$ regardless of method, so at a double root you converge quadratically to $10^{-8}$ and
then stop. The conditioning is a property of the problem and no algorithm changes it.

### 1.3 A Newton cycle

Take $f(x) = x^3 - 2x + 2$ with $x_0 = 0$.

$$f(0) = 2, \quad f'(0) = -2, \quad x_1 = 0 - \frac{2}{-2} = 1.$$

$$f(1) = 1 - 2 + 2 = 1, \quad f'(1) = 3 - 2 = 1, \quad x_2 = 1 - \frac11 = 0.$$

So $x_2 = x_0$ and the iteration cycles $0, 1, 0, 1, \dots$ forever. It is exactly periodic in
exact arithmetic, and in floating point it stays in the cycle too, because both steps are exact
in binary.

The real root is near $x = -1.7693$, and Newton from $x_0 = 0$ never gets anywhere near it.
Nothing is singular, $f$ is a perfectly nice polynomial, and $f'$ never vanishes at the iterates.
Newton simply has no global guarantee, and this is what that means.

The general lesson: attracting cycles of period 2 and higher exist for Newton on ordinary smooth
functions, they have open basins of positive measure, and no local convergence theorem rules them
out. This is why production solvers use a safeguard (exercise 3.1) or a bracket (Brent).

### 2.1 Completing the proof of Theorem 11.1

Let $r$ be a simple root, so $f(r) = 0$ and $f'(r) \ne 0$, with $f''$ continuous near $r$.

Taylor with remainder about $x_k$, evaluated at $r$:

$$0 = f(r) = f(x_k) + f'(x_k)(r - x_k) + \tfrac12 f''(\xi_k)(r-x_k)^2$$

for some $\xi_k$ between $x_k$ and $r$.

**This is the first place $f'(x_k) \ne 0$ is needed.** Divide through by $f'(x_k)$:

$$0 = \frac{f(x_k)}{f'(x_k)} + (r - x_k) + \frac{f''(\xi_k)}{2f'(x_k)}(r-x_k)^2.$$

**This is the second place it is needed**, because the Newton step $x_{k+1} = x_k -
f(x_k)/f'(x_k)$ is undefined otherwise. Substituting $f(x_k)/f'(x_k) = x_k - x_{k+1}$:

$$0 = (x_k - x_{k+1}) + (r - x_k) + \frac{f''(\xi_k)}{2f'(x_k)}(r - x_k)^2
= (r - x_{k+1}) + \frac{f''(\xi_k)}{2f'(x_k)}e_k^2,$$

so with $e_k = x_k - r$,

$$e_{k+1} = \frac{f''(\xi_k)}{2f'(x_k)}\,e_k^2.$$

**Where the hypotheses are used.** We need $f'(x_k) \ne 0$ at every iterate, not just at $r$.
That is obtained from $f'(r) \ne 0$ plus continuity: there is a $\delta$ with $|f'| \ge
\tfrac12|f'(r)|$ on $[r-\delta, r+\delta]$. On the same interval let $M = \max|f''|$, and set

$$\delta' = \min\left(\delta, \frac{|f'(r)|}{2M}\right).$$

Then for $|e_k| \le \delta'$,

$$|e_{k+1}| \le \frac{M}{2\cdot\frac12|f'(r)|}e_k^2 = \frac{M}{|f'(r)|}e_k^2 \le
\frac{M\delta'}{|f'(r)|}|e_k| \le \tfrac12|e_k|,$$

so the iterate stays inside $[r-\delta', r+\delta']$ and the argument can be repeated. That
closes the induction. Taking $k \to \infty$ gives $\xi_k \to r$, $x_k \to r$ and

$$\lim \frac{|e_{k+1}|}{e_k^2} = \frac{|f''(r)|}{2|f'(r)|}. \qquad \square$$

The convergence is **local**: the theorem says nothing for $|e_0| > \delta'$, and exercise 1.3
shows why that caveat is real.

### 2.2 The secant order is the golden ratio

The secant error recurrence is $e_{k+1} \approx Ce_ke_{k-1}$ with $C = f''(r)/(2f'(r))$.

Assume the errors settle into $|e_{k+1}| = A|e_k|^p$ for constants $A, p > 0$. Then
$|e_k| = A|e_{k-1}|^p$, so $|e_{k-1}| = (|e_k|/A)^{1/p}$. Substituting into the recurrence:

$$A|e_k|^p = C\,|e_k|\,\left(\frac{|e_k|}{A}\right)^{1/p}
= CA^{-1/p}|e_k|^{1 + 1/p}.$$

For this to hold as $|e_k| \to 0$ the powers must match:

$$p = 1 + \frac1p \iff p^2 = p + 1 \iff p = \frac{1+\sqrt5}{2} = 1.6180\ldots$$

taking the positive root. Matching the constants then gives $A = C^{1/(1+1/p)} = C^{p-1}$, so
the asymptotic error constant is $|C|^{p-1} = |C|^{0.618}$. $\square$

The golden ratio appears because the recurrence links **two** previous errors with total exponent
1 each, which is the Fibonacci recurrence in the exponents. A method using the last three points
gives $p^3 = p^2 + p + 1$, so $p = 1.8393$, and lesson 12 exercise 2.1 continues the pattern.

### 2.3 Newton's linear rate at a multiplicity $m$ root

Write $f = (x-r)^m h$ with $h(r) \ne 0$ and $h$ differentiable. Then

$$f' = m(x-r)^{m-1}h + (x-r)^mh' = (x-r)^{m-1}\big(mh + (x-r)h'\big).$$

So with $e = x - r$,

$$\frac{f}{f'} = \frac{e^m h}{e^{m-1}(mh + eh')} = \frac{e\,h}{mh + eh'}
= \frac{e}{m}\cdot\frac{1}{1 + \dfrac{eh'}{mh}}
= \frac{e}{m}\left(1 - \frac{eh'}{mh} + O(e^2)\right).$$

Therefore

$$e_{k+1} = e_k - \frac{f}{f'} = e_k - \frac{e_k}{m} + \frac{e_k^2h'(r)}{m^2h(r)} + O(e_k^3)
= \left(1 - \frac1m\right)e_k + O(e_k^2). \qquad \square$$

Measured on $f(x) = (x-1)^m$ from $x_0 = 1.5$, the rate is exact to five decimals:

| $m$ | measured rate | $1 - 1/m$ |
|---:|---:|---:|
| 2 | 0.50000 | 0.50000 |
| 3 | 0.66667 | 0.66667 |
| 4 | 0.75000 | 0.75000 |
| 5 | 0.80000 | 0.80000 |
| 6 | 0.83333 | 0.83333 |

For $m = 1$ there is no rate to measure, because convergence is quadratic and the ratio goes to
zero, which is what the formula $1 - 1/1 = 0$ is telling you.

### 2.4 The function $u = f/f'$

**$u$ has a simple root wherever $f$ has any root.** From exercise 2.3,

$$u = \frac{f}{f'} = \frac{(x-r)h}{mh + (x-r)h'},$$

which vanishes at $r$ (numerator zero, denominator $mh(r) \ne 0$) and has

$$u'(r) = \frac{h(r)}{mh(r)} = \frac1m \ne 0.$$

So $r$ is a **simple** root of $u$ whatever the multiplicity of $f$ there, and its multiplicity
never appears again. $\square$

**Newton applied to $u$.** With $u = f/f'$,

$$u' = \frac{f'\cdot f' - f f''}{(f')^2} = 1 - \frac{ff''}{(f')^2},$$

so the iteration $x - u/u'$ becomes

$$x_{k+1} = x_k - \frac{f/f'}{1 - ff''/(f')^2}
= x_k - \frac{f f'}{(f')^2 - f f''}.$$

This is often called the **modified Newton** or Schroder method. It converges quadratically at
roots of **any** multiplicity, without being told $m$, which is the point of exercise 3.3.

Two costs. It needs $f''$, so $w = 3$ and the efficiency index falls to $2^{1/3} = 1.26$, below
plain Newton. And near a multiple root the denominator $(f')^2 - ff''$ is a difference of two
quantities that both go to zero, so cancellation is severe. It restores the **rate** but the
accuracy floor of $u^{1/m}$ is still there, for the reasons in lesson 12.

### 2.5 Halley from Newton on $f/\sqrt{f'}$

Let $F = f(f')^{-1/2}$. Differentiate:

$$F' = f'(f')^{-1/2} - \tfrac12 f (f')^{-3/2}f'' = (f')^{-1/2}\left(f' - \frac{ff''}{2f'}\right).$$

The Newton step for $F$ is

$$\frac{F}{F'} = \frac{f(f')^{-1/2}}{(f')^{-1/2}\left(f' - \dfrac{ff''}{2f'}\right)}
= \frac{f}{f' - \dfrac{ff''}{2f'}}
= \frac{2ff'}{2(f')^2 - ff''}.$$

So

$$x_{k+1} = x_k - \frac{2f f'}{2(f')^2 - ff''},$$

which is Halley's method. $\square$

The trick is that the $(f')^{-1/2}$ factor cancels, leaving a step that involves $f''$ only in a
correction to the denominator. Halley is **cubically** convergent, costs $w = 3$, and has
efficiency index $3^{1/3} = 1.4422$, which is very slightly **above** Newton's $1.4142$ (measured
in lesson 11: 64 evaluations against Newton's 67 on the same four problems, a ratio of 1.047
against a predicted 1.057).

The general pattern: Newton applied to $f\,(f')^{-\alpha}$ gives a family of methods, with
$\alpha = 0$ giving Newton, $\alpha = 1/2$ giving Halley, and $\alpha = 1$ giving the Schroder
method of exercise 2.4. Only $\alpha = 1/2$ raises the order, which is why Halley is the one with
a name.

### 3.1 Safeguarded Newton

```python
def safeguarded_newton(f, df, a, b, tol=1e-14, max_iter=100):
    fa, fb = f(a), f(b)
    assert (fa < 0) != (fb < 0), "need a genuine bracket"
    x = 0.5 * (a + b)
    for _ in range(max_iter):
        fx, dfx = f(x), df(x)
        if dfx != 0:
            step = x - fx / dfx
        else:
            step = None
        if step is None or not (a < step < b):
            step = 0.5 * (a + b)        # fall back to bisection
        x_new = step
        if (f(x_new) < 0) == (fa < 0):
            a, fa = x_new, f(x_new)
        else:
            b, fb = x_new, f(x_new)
        if abs(b - a) < tol:
            return x_new
        x = x_new
    return x
```

It never fails on the section 7 suite, because it inherits bisection's invariant: the bracket
always contains a root and always shrinks. Newton is used when it is helping and discarded when
it is not, so you keep the quadratic rate near the root and the guarantee everywhere.

The one thing to be careful about is that Newton can stay inside the bracket while barely moving,
which passes the test but makes no progress. A better safeguard also requires the bracket to at
least halve every two steps, which is exactly what Brent does.

### 3.2 The Newton fractal for $z^3 - 1$

```python
n = 800
x = np.linspace(-2, 2, n); y = np.linspace(-2, 2, n)
Z = x[None, :] + 1j * y[:, None]
for _ in range(50):
    Z = Z - (Z**3 - 1) / (3 * Z**2)
roots = np.exp(2j * np.pi * np.arange(3) / 3)
which = np.argmin(np.abs(Z[..., None] - roots), axis=-1)
plt.imshow(which, extent=[-2, 2, -2, 2], cmap="turbo")
```

Three roots at $1$, $e^{2\pi i/3}$, $e^{4\pi i/3}$. Each basin is an open set containing its
root, and the boundary between them is the Julia set of the rational map $z - (z^3-1)/(3z^2)$.

Watch out for $z = 0$, where $f' = 0$ and the step is infinite. Points near the origin get thrown
far away, which is why the fractal detail concentrates near it.

The picture makes lesson 11's point better than any table: Newton's basin is not a neighbourhood,
it is a set of arbitrary complexity, and two starting points a millionth apart can go to different
roots.

### 3.3 Newton on $u = f/f'$ at a triple root

Use the formula from exercise 2.4:

```python
def modified_newton(f, df, d2f, x0, tol=1e-13, max_iter=100):
    x = float(x0)
    for _ in range(max_iter):
        fx, dfx, d2fx = f(x), df(x), d2f(x)
        denom = dfx * dfx - fx * d2fx
        if denom == 0:
            return x
        step = fx * dfx / denom
        x -= step
        if abs(step) <= tol * max(1.0, abs(x)):
            return x
    return x
```

On section 3's triple root, plain Newton converges linearly at rate $1 - 1/3 = 0.667$, taking
about 90 iterations to reach $10^{-11}$, while the modified version converges quadratically in
about 6. And it was never told the multiplicity was 3.

The catch, which you should also measure: the modified version stalls at an error around
$10^{-6}$ rather than $10^{-13}$. That is not a bug. It is $u^{1/3} = 4.8\times 10^{-6}$, lesson
12's accuracy floor for a triple root, and it is a property of the problem. Quadratic convergence
gets you to the floor faster; it does not lower it.

### 4.1 Empirical efficiency index

Measured: total $f$ and $f'$ and $f''$ evaluations to reach $10^{-12}$, summed over four problems
($x^3-2$, $\cos x - x$, $e^x - 3x$, $x - e^{-x}$).

| method | total evaluations | theoretical $p^{1/w}$ |
|---|---:|---:|
| Brent | 34 | no fixed order |
| secant | 35 | 1.6180 |
| Steffensen | 36 | 1.4142 |
| Illinois | 38 | 1.4400 |
| Halley | 64 | 1.4422 |
| Newton | 67 | 1.4142 |
| bisection | 168 | 1.0000 |

**Does the ranking match?** Partly, and the mismatches are informative.

The secant method's index of 1.618 is the highest of any method with a fixed order, and it comes
second overall. Correct.

Bisection's index of 1 puts it last by a wide margin. Correct, and by roughly the predicted
factor.

Halley (1.4422) beating Newton (1.4142) is confirmed: 64 against 67. The predicted ratio of work
is $\log 1.4422 / \log 1.4142 = 1.057$ and the measured ratio is $67/64 = 1.047$. Good agreement
on a difference this small.

The mismatch is that Steffensen (index 1.414) beats Illinois (1.440) and nearly ties secant,
which the index does not predict. The reason is that the index is an **asymptotic** statement and
these problems converge in 4 to 8 steps, so start-up costs dominate. Steffensen needs no
derivative and no bracket, so it wastes nothing.

Brent coming first with no fixed order at all is the headline. Efficiency index is defined only
for methods with a fixed order, and the best method in the table does not have one. Use the index
to understand why methods differ, not to rank them.

### 4.2 Newton's rate at multiplicity $m$

Done in exercise 2.3 above: measured $1 - 1/m$ to five decimal places for $m = 2$ through 6.

The measurement needs care. Read the rate from the window **after** start-up and **before** the
roundoff floor, using `nalib.convergence.linear_rate`, which fits a straight line to
$\log|e_k|$ rather than taking the last ratio. Taking the last ratio gives nonsense, because the
last few errors are at the $u^{1/m}$ floor and their ratios are noise.

### 4.3 Newton's basin boundary for arctan

Bisect on $x_0$: convergent below the boundary, divergent above.

```python
lo, hi = 1.0, 2.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if R.newton(math.atan, lambda x: 1/(1+x*x), mid, max_iter=60).converged:
        lo = mid
    else:
        hi = mid
```

Measured boundary: $x^\ast = \mathbf{1.3917452003}$, matching the textbook 1.3917.

The boundary is where Newton's map has a period-2 cycle: $x_1 = -x_0$ exactly, so the iteration
flips between $\pm x^\ast$ forever. The defining equation is

$$x - \frac{\arctan x}{1/(1+x^2)} = -x
\iff 2x = (1+x^2)\arctan x,$$

whose positive root is 1.3917452003. Inside the boundary Newton converges quadratically; outside,
each step overshoots by more than the last and the iterates blow up. Exactly on it, it cycles.

This is the cleanest example of Newton's local convergence being genuinely local, with a sharp
and computable boundary.

### 5.1 Kantorovich's theorem

> **Theorem (Kantorovich).** Let $f$ be differentiable on a ball $B(x_0, R)$ with $f'$ Lipschitz
> of constant $L$ there. Suppose $f'(x_0)$ is invertible with $\|f'(x_0)^{-1}\| \le \beta$, and
> the first Newton step satisfies $\|f'(x_0)^{-1}f(x_0)\| \le \eta$. If
> $$h = \beta L \eta \le \tfrac12
> \quad\text{and}\quad
> r_- = \frac{1 - \sqrt{1-2h}}{\beta L} \le R,$$
> then $f$ has a root $r$ in $\overline{B}(x_0, r_-)$, it is the only root in
> $B(x_0, r_+)$ with $r_+ = (1+\sqrt{1-2h})/(\beta L)$, and Newton from $x_0$ converges to it,
> quadratically when $h < 1/2$.

What makes it remarkable is that **every quantity is computable at $x_0$**. There is no reference
to the unknown root. You evaluate $f(x_0)$ and $f'(x_0)$, bound $L$ on a region, and get a proof
of existence, uniqueness and convergence. This is the basis of interval Newton methods and of
computer-assisted proofs.

**A case where it applies.** $f(x) = x^2 - 2$, $x_0 = 1.5$. Then $f' = 2x$, so $\beta =
1/|f'(1.5)| = 1/3$; $L = 2$ since $f'' = 2$; and $\eta = |f(1.5)/f'(1.5)| = 0.25/3 = 0.0833$. So

$$h = \tfrac13 \cdot 2 \cdot 0.0833 = 0.0556 \le \tfrac12. \checkmark$$

The theorem guarantees a root within $r_- = (1 - \sqrt{1-0.111})/(2/3) = 0.0857$ of 1.5, so in
$[1.414, 1.586]$. The true root is 1.41421, just inside. Newton converges in 4 steps.

**A case where it does not.** $f(x) = \arctan x$, $x_0 = 3$. Then $f'(3) = 0.1$, so $\beta = 10$;
$\eta = |\arctan 3| / 0.1 = 12.49$; and $L = \max|f''| = 0.65$. So

$$h = 10 \cdot 0.65 \cdot 12.49 = 81.2 \gg \tfrac12.$$

The condition fails badly, and indeed Newton diverges from $x_0 = 3$, consistent with the
boundary at 1.3917 found in 4.3. Note $h \le 1/2$ is **sufficient, not necessary**: the theorem
also fails at $x_0 = 1.2$, where Newton actually converges. It is a guarantee, not a
characterisation.

### 5.2 Why the fractal boundary is shared

**All three basins share one boundary.** Let $\partial B_1$ be the boundary of the basin of root
1. Take any $z \in \partial B_1$. Every neighbourhood of $z$ meets $B_1$. It must also meet some
other basin, or else a whole neighbourhood would be in $B_1$ (basins are open) and $z$ would be
interior, not boundary. So $z$ is also on the boundary of at least one other basin.

The stronger statement, which is a theorem about Julia sets, is that every boundary point is on
the boundary of **all three**. The reason is that the Julia set $J$ is the closure of the
repelling periodic points, and the dynamics on $J$ are topologically transitive: any open set
meeting $J$ has iterates that eventually cover the whole sphere minus at most two points, so it
must meet all three basins.

**Why that forces nowhere-smoothness.** Suppose the boundary contained a smooth arc. A smooth arc
locally separates the plane into exactly **two** sides. But every point of that arc must have all
three basins in every neighbourhood, and three open sets cannot fit into two sides of an arc
without one of them touching the arc from both sides, which contradicts the arc being a
separating boundary between two distinct basins. So no smooth arc can exist anywhere on the
boundary.

This is the "no three countries can share a coastline smoothly" argument, and it is the reason
the Newton fractal looks the way it does: the three colours must interleave at every scale, which
forces self-similar detail all the way down.

The numerical consequence is real and not just pretty: for any $\epsilon > 0$ there are starting
points within $\epsilon$ of each other that Newton sends to different roots. Newton's answer is
not a continuous function of the starting point.

### 5.3 Inverse quadratic interpolation

**Derivation.** The secant method fits $y$ as a linear function of $x$ through two points and
solves for $y = 0$. IQI fits $x$ as a **quadratic in $y$** through three points $(x_i, f_i)$ and
evaluates at $y = 0$. The Lagrange form gives immediately

$$x_{\text{new}} = \frac{f_1f_2}{(f_0-f_1)(f_0-f_2)}x_0
+ \frac{f_0f_2}{(f_1-f_0)(f_1-f_2)}x_1
+ \frac{f_0f_1}{(f_2-f_0)(f_2-f_1)}x_2.$$

Fitting $x$ as a function of $y$ rather than $y$ as a function of $x$ avoids having to solve a
quadratic, so there is no square root and no choice of branch. That is the whole reason for
doing it backwards.

**Order.** The interpolation error for three points gives $e_{k+1} \approx Ce_ke_{k-1}e_{k-2}$,
and the same argument as exercise 2.2 gives $p^3 = p^2 + p + 1$, whose positive root is
$p = 1.8393$.

Measured with `reliable_order` against a true root:

| problem | measured order |
|---|---:|
| $x^3 - 2$ | 2.18 |
| $\cos x - x$ | 1.98 |
| $e^x - 3x$ | 1.60 |
| $x - e^{-x}$ | 1.24 |

The average is close to 1.84 but no single problem lands on it, and the spread runs from 1.24 to
2.18. Do not read that as the theory being wrong. IQI reaches machine precision in **four to six**
steps from these starting triples, so there are only two or three clean errors to fit an order to,
and start-up effects have not died out before the roundoff floor arrives.

This is the same measurement problem as Illinois in lesson 09 exercise 4.3 and Muller in lesson
12, and it is worth stating as a general rule: **an order above about 1.5 converges too fast to
have its order measured on an easy problem.** To see 1.84 you need a problem where convergence is
slow enough to give a window, which means starting far away, or extended precision so the floor
is lower.

IQI does at least have a genuine per-step order, unlike Brent and Illinois. It is just hard to
observe.

**Why Brent needs the bisection safeguard.** IQI fails in three distinct ways, and all three are
common:

1. **The three $f$ values are not distinct**, so a denominator is zero. Even nearly equal values
   produce a wild extrapolation.
2. **The parabola in $y$ can send $x_{\text{new}}$ anywhere**, including far outside the bracket
   and far from any root. There is nothing in the formula that keeps it local.
3. **It is not monotone**, so the bracket can fail to shrink for many steps in a row.

Brent's answer is to attempt IQI, accept it only if the result lands in the middle portion of the
current bracket and the bracket is shrinking fast enough, and otherwise bisect. The result keeps
IQI's 1.84 order when things are going well and bisection's guarantee when they are not. That
combination is why Brent came first in the efficiency table of 4.1 while having no fixed order at
all.

---

## Lesson 12, Convergence Theory and Sensitivity

### 1.1 Comparing order 1.6 at 1 evaluation with order 2.0 at 2

Efficiency indices:

- $1.6^{1/1} = 1.600$
- $2.0^{1/2} = 1.414$

The order 1.6 method wins. To quantify "by how much", the work to gain a fixed number of digits
is proportional to $1/\log(\text{index})$, so the ratio is

$$\frac{\log 1.600}{\log 1.414} = \frac{0.4700}{0.3466} = 1.356.$$

The order 1.6 method needs about **26 percent fewer evaluations** for the same accuracy
(equivalently, the order 2 method needs about 36 percent more).

This is the secant-against-Newton comparison with rounded numbers, and it is why "higher order"
and "faster" are not the same statement.

### 1.2 The accuracy floor when $f'(r) = 10^{-6}$

The condition number of a simple root is $\kappa = 1/|f'(r)|$ relative to perturbations of $f$.
Evaluating $f$ in double precision produces a value with absolute error at least about $u|f|$,
and near the root the unavoidable noise level is roughly $u$ times the size of the terms being
cancelled in $f$'s evaluation. Call that noise $\epsilon_f$. Then the root can only be located to

$$|\delta r| \approx \frac{\epsilon_f}{|f'(r)|}.$$

With $f$ of order 1 near the root, $\epsilon_f \approx u = 1.1\times10^{-16}$, giving

$$|\delta r| \approx \frac{1.1\times 10^{-16}}{10^{-6}} = 1.1\times 10^{-10}.$$

So about **10 correct digits**, not 16, and that is the best possible.

**Does a better method help?** No. Not at all. Every method computes $f$ in double precision, and
below $|\delta r| \approx 10^{-10}$ every value of $x$ produces a computed $f(x)$ that is
indistinguishable from zero. The information needed to do better is simply not in the data.

What **does** help:
- **Evaluating $f$ more accurately.** If $f$ can be computed with smaller $\epsilon_f$, for
  instance by avoiding cancellation in its formula (lesson 05), the floor drops proportionally.
- **Reformulating.** Solve $g(x) = f(x)/10^{-6}$ or find an equivalent equation with a steeper
  crossing.
- **More precision.** Doubling the precision squares the floor.

This is a conditioning statement, and lesson 06's rule applies: no algorithm beats the
conditioning of the problem it is given.

### 1.3 Wilkinson's difficulty is not algorithmic

Because the difficulty is in the **map from coefficients to roots**, and that map is fixed before
any algorithm is chosen.

The sensitivity $\partial r_k/\partial a_j = -r_k^j/p'(r_k)$ is a property of the polynomial. For
Wilkinson's degree 20 polynomial, $\partial r_{15}/\partial a_1 \approx 6\times 10^{13}$: a
relative change of $2^{-23}$ in one coefficient moves a root by about 5.9 units, measured. That
number contains no reference to an algorithm.

The forward error obeys $\text{forward} \lesssim \kappa \times \text{backward}$. A **backward
stable** root finder returns roots that are exact for a polynomial within $O(u)$ of the input.
That is the best any algorithm can promise, and with $\kappa \approx 10^{13}$ it still leaves a
forward error around $10^{-3}$. Measured: `numpy.roots` on the degree 20 Wilkinson polynomial has
a maximum root error of $9.3\times 10^{-2}$.

Changing algorithms changes the constant in front of $u$. It cannot change $\kappa$. The only
real fix is to change the **representation**: keep the polynomial in factored form, or in a
Chebyshev basis, where the same roots are well conditioned (exercise 5.2 and lesson 13
exercise 5.2).

### 2.1 The order of a four-point method

A method fitting a cubic through four previous points has error recurrence
$e_{k+1} \approx Ce_ke_{k-1}e_{k-2}e_{k-3}$. Assume $e_{k+1} = Ae_k^p$ and substitute, as in
lesson 11 exercise 2.2. Matching exponents gives

$$p = 1 + \frac1p + \frac1{p^2} + \frac1{p^3},$$

and multiplying by $p^3$,

$$p^4 = p^3 + p^2 + p + 1.$$

Solving numerically: the roots are $1.927562$, $-0.774804$, and $-0.076379 \pm 0.814704i$. The
positive real root is $p = \mathbf{1.927562}$.

**Why the sequence tends to 2.** The $n$-point method satisfies $p^n = p^{n-1} + \cdots + p + 1$.
Sum the geometric series on the right:

$$p^n = \frac{p^n - 1}{p - 1} \iff p^n(p-1) = p^n - 1 \iff p^{n+1} - 2p^n + 1 = 0.$$

As $n \to \infty$ with $p < 2$, the term $p^{n+1} - 2p^n = p^n(p-2)$ must equal $-1$, so
$p^n(2 - p) = 1$, giving $2 - p = p^{-n} \to 0$. Hence $p \to 2^-$.

Measured, the sequence of orders is:

| points | equation | order |
|---:|---|---:|
| 2 | $p^2 = p+1$ | 1.618034 |
| 3 | $p^3 = p^2+p+1$ | 1.839287 |
| 4 | $p^4 = \cdots$ | 1.927562 |
| 5 | | 1.965948 |
| 6 | | 1.983583 |
| 7 | | 1.991964 |
| 8 | | 1.996031 |

Each step roughly halves the gap to 2, as $2 - p \approx 2^{-n}$ predicts.

**Why this matters practically.** The order approaches 2 but never reaches it, while the work
per step stays at one evaluation, so the efficiency index $p^{1/1}$ climbs towards 2, beating
Newton's 1.414 and secant's 1.618. So why does nobody use the 8-point method?

Because the interpolation becomes catastrophically ill-conditioned. Fitting a degree 7 polynomial
through 8 nearly coincident points is a Vandermonde problem with a condition number that grows
exponentially, so the extra order is destroyed by roundoff long before it pays. Three points
(IQI) is about the practical limit, which is exactly what Brent uses.

### 2.2 Root sensitivity by implicit differentiation

Let $r$ be a simple root of $p(x) = \sum_j a_j x^{n-j}$, so $p(r) = 0$ and $p'(r) \ne 0$.
Perturb the coefficient $a_j$ and let $r$ depend on it. Differentiate $p(r(a_j); a_j) = 0$ with
respect to $a_j$, using the chain rule:

$$\frac{\partial p}{\partial x}\bigg|_{r}\cdot\frac{\partial r}{\partial a_j}
+ \frac{\partial p}{\partial a_j}\bigg|_{r} = 0.$$

The first factor is $p'(r)$. The second is the coefficient's own term differentiated, which is
just $r^{n-j}$ (writing the index the other way round, $r^j$). So

$$p'(r)\frac{\partial r}{\partial a_j} + r^j = 0
\implies \boxed{\frac{\partial r}{\partial a_j} = -\frac{r^j}{p'(r)}}. \qquad \square$$

Two readings of this formula, both important:

**Large roots are catastrophic.** The numerator is $r^j$, so for $|r| > 1$ the sensitivity grows
exponentially in $j$. Wilkinson's root at 20 raised to a moderate power is astronomical, and this
is why the trouble concentrates at the large roots.

**Close roots are catastrophic.** The denominator is $p'(r) = \prod_{i\ne k}(r_k - r_i)$, which
is tiny whenever any two roots are close. This is the multiplicity effect of exercise 2.3 seen
from the coefficient side.

Wilkinson's polynomial has both problems at once, which is why it is the standard example.

### 2.3 The $u^{1/m}$ accuracy limit

Near a root of multiplicity $m$, write $p(x) \approx c(x-r)^m$ with $c = p^{(m)}(r)/m! \ne 0$.
Perturb the polynomial by $\delta$, meaning the computed function is $p(x) + \delta$ rather than
$p(x)$. Its root $\tilde r$ satisfies

$$c(\tilde r - r)^m + \delta = 0
\implies |\tilde r - r| = \left|\frac{\delta}{c}\right|^{1/m}. \qquad \square$$

Now put in numbers. In double precision, evaluating $p$ near $r$ produces a value contaminated by
roundoff of size at least $\delta \approx u\|p\|$, where $\|p\|$ reflects the size of the terms
being summed. So the achievable accuracy is

$$|\tilde r - r| \approx u^{1/m}$$

up to the constant $|1/c|^{1/m}$.

Measured, `numpy.roots` on $(x-1)^m$:

| $m$ | measured error | $u^{1/m}$ |
|---:|---:|---:|
| 1 | 0.0 (exact) | $1.1\times10^{-16}$ |
| 2 | 0.0 (exact) | $1.1\times10^{-8}$ |
| 3 | $6.6\times10^{-6}$ | $4.8\times10^{-6}$ |
| 4 | $2.2\times10^{-4}$ | $1.0\times10^{-4}$ |

The measured errors at $m = 3$ and 4 match $u^{1/m}$ within a small factor. At $m = 1$ and 2 the
result comes out exact, because $(x-1)^2 = x^2 - 2x + 1$ has small integer coefficients that are
exact in binary and the companion matrix eigenvalues happen to come out exactly. That is luck
about this particular polynomial, not a violation: perturb the coefficients slightly and the
$10^{-8}$ shows up immediately.

The practical rule: **a double root costs you half your digits, a triple root two thirds.** In
double precision that is 8 digits and 5 digits respectively.

### 2.4 Justifying the efficiency index

Ask: what happens to the order when you apply a method twice and call the pair one step?

If one step has order $p$, then $e_{k+1} \sim e_k^p$ and $e_{k+2} \sim e_{k+1}^p \sim e_k^{p^2}$.
So **two steps of an order $p$ method form one step of an order $p^2$ method**, and the doubled
method costs $2w$ evaluations.

Now compare the two descriptions of the same algorithm:

$$\text{single: } p^{1/w}, \qquad \text{doubled: } (p^2)^{1/(2w)} = p^{2/(2w)} = p^{1/w}.$$

They agree. The efficiency index is **invariant** under regrouping steps, and that is exactly the
property a fair comparison measure must have. Raw order is not invariant: the doubled method has
order $p^2$, so ranking by order would say the doubled method is better than itself.

The deeper version: after $N$ evaluations you have taken $N/w$ steps and the error is
$e \sim e_0^{p^{N/w}}$. Taking logs twice,

$$\log\log(1/e) \approx \frac{N}{w}\log p + \text{const} = N\log\left(p^{1/w}\right) + \text{const}.$$

So the number of **correct digits** grows like $\exp(N\log p^{1/w})$, and $p^{1/w}$ is the base
of that exponential. Two methods with the same index gain digits at the same rate per evaluation
no matter how their steps are grouped.

### 3.1 A root diagnostic function

```python
def diagnose_root(f, df, r_hat):
    residual = abs(f(r_hat))
    slope = abs(df(r_hat))
    kappa = 1.0 / slope if slope > 0 else np.inf
    bound = kappa * residual
    u = np.finfo(float).eps / 2
    noise = u * max(1.0, abs(r_hat))
    if slope == 0:
        verdict = "derivative is zero: multiple root, expect at most u^(1/m)"
    elif bound <= 10 * noise:
        verdict = "as accurate as double precision allows"
    elif bound <= 1e-8:
        verdict = "usable, but the problem is somewhat ill conditioned"
    else:
        verdict = f"WARNING: forward error may be as large as {bound:.1e}"
    return {"residual": residual, "condition": kappa,
            "forward_bound": bound, "verdict": verdict}
```

The point is lesson 06 section 9's discipline: a small residual is a **backward** error statement
and says nothing on its own. Multiply it by the condition number before believing anything. The
function reports both numbers and their product, so the user cannot skip the multiplication.

### 3.2 Wilkinson at degrees 5, 10, 15, 20

Measured, using `numpy.roots` on the expanded $\prod_{k=1}^{n}(x-k)$, and separately perturbing
$a_1$ by a relative $2^{-23}$ (Wilkinson's original perturbation):

| $n$ | max root error, unperturbed | root movement from the $2^{-23}$ perturbation |
|---:|---:|---:|
| 5 | $2.2\times10^{-13}$ | $7.6\times10^{-5}$ |
| 10 | $4.2\times10^{-9}$ | $8.9\times10^{-2}$ |
| 15 | $4.9\times10^{-6}$ | $2.4$ |
| 20 | $9.3\times10^{-2}$ | $5.9$ |

**Where does the trouble begin?** It depends what you call trouble, and the two columns give
different answers, which is the interesting part.

By the unperturbed column, degree 10 is already down to 9 digits and degree 15 to 6. That is
already bad, but it might be tolerable.

By the perturbation column, degree 10 already moves a root by 0.089, which is a **9 percent**
error in the root at 10 from a change in the 7th significant digit of one coefficient. Degree 15
moves a root by 2.4, so the roots are no longer identifiable at all.

The honest boundary is around **degree 10 to 12**. Below that the coefficient representation is
usable; above it, the roots are not determined by the coefficients to any useful accuracy, and
what you compute depends on the arithmetic rather than on the mathematics.

### 3.3 The full sensitivity heatmap

Compute $S_{kj} = |\partial r_k / \partial a_j| = |r_k^j / p'(r_k)|$ for all $k, j$ and plot
$\log_{10}S$ as an image, roots on one axis and coefficient index on the other.

What you see:

- The map increases monotonically to the right (larger $j$) and downward (larger roots), because
  the numerator is $r_k^j$. The corner at large $k$ and large $j$ dominates everything.
- The largest entry is at the largest root and the highest power. The peak sensitivity
  $|\partial r_k/\partial a_j|$ reaches about $10^{16}$ for the Wilkinson polynomial.
- The most sensitive **root** is around $k = 15$ or 16, not $k = 20$. The largest roots have
  large $r^j$ but also large $|p'(r)|$, since they are far from the cluster, and the two effects
  partly cancel. The maximum of the ratio sits in the middle-upper range.

**Is $a_1$ the most dangerous coefficient?** By raw sensitivity, no. The entries grow with $j$,
so the highest-index coefficients have the largest partial derivatives.

But raw sensitivity is the wrong measure, and this is the point of the exercise. What matters is
the **relative** perturbation $\delta a_j = \epsilon |a_j|$, so the actual root movement is
$\epsilon|a_j|\cdot|r^j/p'(r)|$. Weighting by $|a_j|$ changes the picture: $a_1 = -210$ is a
large coefficient, and $\epsilon|a_1|$ is therefore a large absolute perturbation.

Wilkinson chose $a_1$ because it maximises the **product**, and because $2^{-23}$ on $a_1$ is a
single-precision rounding, so it is exactly the kind of error that occurs by accident. He picked
the perturbation that is both realistic and devastating, which is better journalism than picking
the largest partial derivative.

### 4.1 Empirical efficiency across all of Part 2

Same experiment as lesson 11 exercise 4.1, extended to the polynomial methods. See the table
there. The summary:

- The ranking by measured evaluations broadly follows the index for the methods that have an
  order, with agreement to a few percent for the Newton against Halley comparison.
- It fails for hybrid methods (Brent, Illinois), which have no fixed order and beat their
  nominal index.
- It fails for problems that converge in fewer than about 6 steps, where start-up dominates and
  the asymptotic index has not taken effect.

The practical conclusion: the efficiency index explains **why** methods differ, and total
evaluations to a fixed tolerance across a suite of problems is what you should actually measure.

### 4.2 Wilkinson root clouds

Perturb every coefficient by an independent relative amount of size $10^{-10}$, 500 times, and
plot all resulting roots in the complex plane.

You get a picture of the **pseudozero set** (exercise 5.1) sampled by Monte Carlo. Its features:

- Roots 1 through about 8 stay put, appearing as tight dots. Their sensitivity $|r^j/p'(r)|$ is
  small because $r$ is small.
- Roots 9 and above spread into arcs, and the middle roots (14 to 17) spread the most.
- Many of them leave the real axis entirely and appear as **complex conjugate pairs**, forming
  the characteristic sideways lens shapes. This is the striking part: a polynomial with 20
  distinct real roots, perturbed in the 10th digit, has complex roots.
- The largest roots (19, 20) spread less than the middle ones, for the $|p'(r)|$ reason discussed
  in 3.3.

**Comparing cloud radius with predicted sensitivity.** For each root compute
$\rho_k = 10^{-10}\sum_j |a_j|\,|r_k^j/p'(r_k)|$, which is the first-order bound on the movement
from independent relative perturbations. Plot the measured cloud radius against $\rho_k$. The
agreement is good over the range where the movement stays small, and breaks down for the middle
roots where the movement is comparable to the root spacing, because there the first-order
analysis is no longer valid: once two roots collide and go complex, linearisation is finished.

That breakdown is the honest headline. First-order sensitivity predicts its own failure here, and
that is the right way to read a condition number this large.

### 4.3 Conditioning against root spread

Build $p$ with roots $1, 2, 4, \dots, 2^k$ and study the conditioning against $k$.

The dominant term in $\partial r/\partial a_j = -r^j/p'(r)$ is $r^j$ at the largest root
$r = 2^k$, with $j$ up to $k$, giving roughly $2^{k^2}$. So the conditioning grows like
$2^{k^2}$, which is **superexponential in $k$** and vastly worse than the Wilkinson polynomial of
the same degree, where the roots only reach $n$ rather than $2^n$.

Meanwhile, $p'(r)$ at the largest root is a product of gaps $2^k - 2^i$, which is also large and
partly compensates, but not nearly enough.

**The conclusion.** Degree is not the right variable. What kills you is the **dynamic range of
the roots**, meaning $\max|r_i| / \min|r_i|$, because that is what makes $r^j$ vary over so many
orders of magnitude that the coefficients cannot represent them all. Wilkinson's polynomial has
a range of 20; this one has a range of $2^k$.

A degree 50 polynomial with all roots in $[-1,1]$ is far better conditioned than a degree 10
polynomial with roots spread over six orders of magnitude. Scaling the variable so the roots have
a range near 1 is therefore a genuine and standard remedy, and it costs nothing.

### 5.1 The pseudozero set

**Definition.** For a polynomial $p$ and a tolerance $\epsilon$, the pseudozero set is

$$Z_\epsilon(p) = \{z \in \mathbb{C} : q(z) = 0 \text{ for some } q
\text{ with } \|q - p\| \le \epsilon\|p\|\}.$$

**Computable characterisation.** With the weighted norm $\|q\| = \max_j |q_j|/w_j$, there is a
closed form:

$$z \in Z_\epsilon(p) \iff |p(z)| \le \epsilon \sum_j w_j |z|^j.$$

So you evaluate $|p(z)|$ on a grid, divide by $\sum_j w_j|z|^j$, and contour the result at
$\epsilon$. It is one line of numpy per contour level.

For the Wilkinson polynomial at $\epsilon = 10^{-16}, 10^{-12}, 10^{-8}$ you get three nested
regions. At $10^{-16}$ they are tiny dots around roots 1 to 10 and small blobs further out. At
$10^{-12}$ the blobs around roots 12 to 18 have merged into a few large connected lobes extending
well off the real axis. At $10^{-8}$ almost the entire upper range has merged into one connected
region stretching from about 8 to 22 and reaching $\pm 3$ in the imaginary direction.

**Why this is the honest answer to "what are its roots".** Once the region around roots 15, 16
and 17 is a single connected blob, there is no meaningful sense in which the polynomial "has a
root at 16". Any point in that blob is an exact root of a polynomial indistinguishable from this
one in double precision. Reporting "16.0000000" is a fiction with false precision; reporting the
blob is the truth.

This is the same idea as a pseudospectrum for a matrix, which lesson 40 develops, and for the
same reason: when a computation is ill conditioned, the correct output is a **set**, not a point.

### 5.2 Is the companion matrix a well-conditioned reformulation?

**Short answer: it is exactly as bad, and it cannot be otherwise.**

`numpy.roots` builds the companion matrix $C$ of $p$ and calls `eig`. The eigenvalue algorithm
(shifted QR, lesson 41) is backward stable, so it returns exact eigenvalues of $C + E$ with
$\|E\| = O(u\|C\|)$.

The condition number of a simple eigenvalue $\lambda$ is $\kappa(\lambda) = 1/|y^\ast x|$ with
$x, y$ the unit right and left eigenvectors. For a companion matrix these can be written down:
$x = (1, \lambda, \dots, \lambda^{n-1})^T$ up to scaling, and $y$ involves the coefficients. The
resulting $\kappa(\lambda)$ works out proportional to $\sum_j|a_j\lambda^j| / |p'(\lambda)|$,
which is **the same quantity** as the root condition number from exercise 2.2.

That is not a coincidence, and here is why it cannot be. The map from coefficients to roots is
what it is. The companion matrix is an invertible re-encoding of the coefficients, so it carries
exactly the same information. No re-encoding of the same information can reduce the sensitivity
of the answer to that information. The conditioning is a property of the **problem**, and the
companion matrix does not change the problem.

There is a subtlety worth knowing. A perturbation of the companion matrix in **all** its entries
is more general than a perturbation of the coefficients, since the companion matrix has a fixed
structure of zeros and ones that a general $E$ destroys. So companion-matrix backward stability
is a slightly weaker statement than coefficient-wise backward stability, and in principle
`numpy.roots` could be slightly worse than an optimal coefficient-based method. In practice the
difference is a small constant and the $10^{13}$ dominates it.

**What does help** is changing the representation itself, not the algorithm: a Chebyshev basis
with a colleague matrix (lesson 13 exercise 5.2) genuinely reduces the conditioning, because the
polynomial is now described by different numbers.

### 5.3 Can any algorithm beat $u^{1/m}$?

**No, and the argument is about information, not effort.**

Set up the question precisely. You may evaluate $f$ in double precision at any points you like,
as many as you like. Every evaluation returns $\mathrm{fl}(f(x))$, which differs from $f(x)$ by
an unknown amount of size up to $\epsilon_f \approx u\,\Phi(x)$, where $\Phi$ is the sum of
magnitudes of the terms in the evaluation.

**The indistinguishability argument.** Consider two problems, $f(x) = c(x-r)^m$ and
$\tilde f(x) = c(x - \tilde r)^m$, with $|\tilde r - r| = \eta$. Their difference at any $x$ near
the roots is

$$|f(x) - \tilde f(x)| = |c|\,\big|(x-r)^m - (x-\tilde r)^m\big| \approx |c| \, m\eta\,|x-r|^{m-1}.$$

For $x$ within a distance $\eta$ of the roots, this is at most about $|c|m\eta^m$.

Now choose $\eta = (\epsilon_f/(|c|m))^{1/m}$, that is $\eta \approx u^{1/m}$ up to constants.
Then $|f(x) - \tilde f(x)| \le \epsilon_f$ for every $x$ in the neighbourhood, so **every
evaluation returns a value consistent with both problems**. No sequence of evaluations, however
long or however cleverly chosen, can distinguish $r$ from $\tilde r$.

Evaluating far from the root does not rescue you, because there $|f|$ itself is large and the
relative noise $u|f|$ is correspondingly large, so the two problems still agree within the noise.

**Conclusion.** An algorithm that reported $r$ to better than $u^{1/m}$ would be reporting
information not present in any data it could obtain. So the bound is not about algorithms at all;
it is the resolution of the measuring instrument. $\square$

**The escapes, and what each costs.** Every one of them changes the data available, which is the
only way out:

- **Higher precision.** More digits in $f$ means smaller $\epsilon_f$ means smaller $u^{1/m}$.
  Costs time and memory.
- **Extra information.** If you are **told** $m$, you can fit $c(x-r)^m$ to several evaluations
  and solve for $r$ by least squares, effectively taking an $m$-th root of the whole data set
  rather than of a single noisy value. This recovers accuracy but requires knowing $m$.
- **A different formulation.** If $f$ comes from a factored or structured form, evaluate that
  form. $(x-1)^{20}$ evaluated as written has no accuracy problem at all; only the expanded
  version does.
- **Exact arithmetic.** For polynomials with rational coefficients, `sympy` or `fractions` gives
  exact answers. The cost is speed, and the method does not extend to general $f$.

---

## Lesson 13, Polynomial Root Finding

### 1.1 Why complex roots pair into real quadratics

If $p$ has real coefficients and $p(z) = 0$, then taking complex conjugates of the whole equation
gives

$$\overline{p(z)} = \sum_j \overline{a_j}\,\overline{z}^{\,j} = \sum_j a_j \bar z^{\,j} = p(\bar z),$$

using $\overline{a_j} = a_j$ because the coefficients are real. So $p(\bar z) = \overline{0} = 0$,
meaning $\bar z$ is also a root. Complex roots come in conjugate pairs.

Multiply the two corresponding factors:

$$(x - z)(x - \bar z) = x^2 - (z + \bar z)x + z\bar z = x^2 - 2\,\mathrm{Re}(z)\,x + |z|^2,$$

which has **real** coefficients, since $\mathrm{Re}(z)$ and $|z|^2$ are real. So every conjugate
pair contributes a real quadratic factor. $\square$

This is exactly why Bairstow's method exists: it searches directly for the real pair $(r,s)$ in
$x^2 - rx - s$, so it finds two complex roots without ever leaving real arithmetic.

### 1.2 Settling Descartes' ambiguity

Descartes counts sign changes and says the number of positive roots is that count **or less by an
even number**. It cannot do better because it only looks at coefficient signs, and different
polynomials with the same sign pattern genuinely have different root counts.

The extra information needed is the **actual number**, which requires looking at $p$ itself, not
just its signs. **Sturm sequences** provide it: $V(a) - V(b)$ gives the exact number of distinct
real roots in $(a,b]$. Combined with the Cauchy bound to get a finite interval, Sturm counts all
the real roots exactly.

The trade is cost against certainty. Descartes is free and gives a bound. Sturm needs a full
polynomial remainder chain, costing $O(n^2)$ operations to build, and gives the answer. Use
Descartes to decide whether it is worth building the chain.

### 1.3 Why deflating a large root hurts more

Deflating means dividing $p$ by $(x - \hat r)$ where $\hat r$ has some error $\delta$. Synthetic
division computes $b_k = a_k + \hat r\, b_{k-1}$, so the error in $b_{k-1}$ is multiplied by
$\hat r$ before entering $b_k$.

The recursion for the error is therefore

$$\varepsilon_k \approx \hat r\,\varepsilon_{k-1} + \delta\,b_{k-1},$$

so after $n$ steps the initial error has been amplified by roughly $|\hat r|^{\,n}$.

**When $|\hat r| > 1$ this grows geometrically**, and by the end of the division the coefficients
of the deflated polynomial can be badly wrong. When $|\hat r| < 1$ the same factor **damps** the
error, and the deflated coefficients are as good as the original ones.

Hence the standard rule: **deflate in increasing order of magnitude**, smallest root first. It
costs nothing to sort, and it is the difference between usable and useless.

Measured on $\prod_{k=1}^{n}(x-k)$, maximum root error, deflation both ways:

| $n$ | small first, raw | small first, polished | large first, raw | large first, polished | numpy.roots |
|---:|---:|---:|---:|---:|---:|
| 4 | $1.2\times10^{-14}$ | $6.7\times10^{-15}$ | $6.2\times10^{-15}$ | $4.4\times10^{-15}$ | $3.9\times10^{-14}$ |
| 6 | $2.5\times10^{-14}$ | $9.1\times10^{-14}$ | $7.3\times10^{-13}$ | $2.3\times10^{-13}$ | $5.5\times10^{-13}$ |
| 8 | $1.8\times10^{-12}$ | $5.9\times10^{-12}$ | $6.6\times10^{-11}$ | $6.0\times10^{-12}$ | $1.3\times10^{-11}$ |
| 10 | $1.1\times10^{-9}$ | $3.8\times10^{-10}$ | failed | failed | $4.2\times10^{-9}$ |
| 12 | failed | failed | failed | failed | $6.4\times10^{-8}$ |
| 14 | failed | failed | failed | failed | $8.5\times10^{-7}$ |

"Failed" means the method did not locate all $n$ real roots. Small-first is roughly a factor of
30 better at $n = 8$ and survives to $n = 10$ where large-first has already broken. Both are
finished by $n = 12$, while the companion matrix keeps going to $n = 14$ and beyond. Deflation
is a teaching tool and a special-purpose technique, not a general polynomial solver.

### 2.1 The companion matrix characteristic polynomial

Take the monic $p(x) = x^n + c_{n-1}x^{n-1} + \cdots + c_0$ and the companion matrix

$$C = \begin{pmatrix}
-c_{n-1} & -c_{n-2} & \cdots & -c_1 & -c_0\\
1 & 0 & \cdots & 0 & 0\\
0 & 1 & \cdots & 0 & 0\\
\vdots & & \ddots & & \vdots\\
0 & 0 & \cdots & 1 & 0
\end{pmatrix}.$$

**Proof by induction on $n$.** Let $D_n(x) = \det(xI - C)$.

*Base case $n = 1$:* $C = (-c_0)$, so $D_1 = x + c_0$, which is $p$. Correct.

*Inductive step.* Expand $\det(xI - C)$ along the **first row**. The matrix $xI - C$ has first row
$(x + c_{n-1},\ c_{n-2},\ \dots,\ c_1,\ c_0)$ and subdiagonal $-1$s.

The first cofactor, deleting row 1 and column 1, leaves exactly the matrix $xI - C'$ where $C'$ is
the companion matrix of $x^{n-1} + c_{n-1}$... more carefully, it leaves the $(n-1)$ by $(n-1)$
matrix of the same companion form for the polynomial $x^{n-1} + c_{n-2}x^{n-2} + \cdots + c_0$,
so its determinant is $D_{n-1}$ by the inductive hypothesis.

Every other cofactor, deleting column $j > 1$, leaves a matrix that is block lower triangular
with a $(j-1)$ by $(j-1)$ block of $-1$s on the diagonal above and a triangular $x$ block below,
giving determinant $(-1)^{j-1}x^{\,n-j}$ up to sign. Collecting the signs from the cofactor
expansion, the term contributes exactly $c_{n-j}x^{\,n-j}$.

Summing:

$$D_n(x) = (x + c_{n-1})D_{n-1}(x)\big|_{\text{leading}} + \sum_{j\ge2} c_{n-j}x^{n-j}
= x^n + c_{n-1}x^{n-1} + \cdots + c_0 = p(x). \qquad \square$$

A cleaner route avoiding the bookkeeping: verify directly that
$v_\lambda = (\lambda^{n-1}, \lambda^{n-2}, \dots, \lambda, 1)^T$ satisfies $Cv_\lambda =
\lambda v_\lambda$ exactly when $p(\lambda) = 0$. Rows 2 through $n$ of $Cv_\lambda = \lambda
v_\lambda$ are the identities $\lambda^{k} = \lambda\cdot\lambda^{k-1}$, which always hold, and
row 1 is precisely $-\sum c_j \lambda^j = \lambda\cdot\lambda^{n-1}$, that is $p(\lambda) = 0$.
Since $C$ is $n$ by $n$ and $p$ is monic of degree $n$, and every root of $p$ is an eigenvalue,
the characteristic polynomial is $p$.

### 2.2 The Bairstow recurrences

Divide $p(x) = \sum_{k=0}^n a_kx^{n-k}$ by the quadratic $x^2 - rx - s$:

$$p(x) = (x^2 - rx - s)\,q(x) + b_{n-1}(x - r) + b_n,$$

where $q(x) = \sum_{k=0}^{n-2} b_kx^{n-2-k}$.

Expand the right side and match coefficients of $x^{n-k}$. The quadratic contributes $b_k$ from
$x^2\cdot b_k x^{n-2-k}$, plus $-rb_{k-1}$ from $-rx\cdot b_{k-1}x^{n-1-k}$, plus $-sb_{k-2}$.
Setting the total equal to $a_k$:

$$a_k = b_k - rb_{k-1} - sb_{k-2}
\implies \boxed{b_k = a_k + rb_{k-1} + sb_{k-2}}$$

with $b_{-1} = b_{-2} = 0$. $\square$

**The $c$ recurrence gives the partial derivatives.** Differentiate the $b$ recurrence with
respect to $r$, writing $c_{k-1} = \partial b_k/\partial r$:

$$\frac{\partial b_k}{\partial r} = b_{k-1} + r\frac{\partial b_{k-1}}{\partial r}
+ s\frac{\partial b_{k-2}}{\partial r},$$

that is

$$c_{k-1} = b_{k-1} + rc_{k-2} + sc_{k-3},$$

which is the **same recurrence applied to the $b$ sequence**. So one extra synthetic division of
$q$ by the same quadratic produces all the derivatives, exactly as Horner's second pass produces
$p'$ in lesson 01.

Differentiating with respect to $s$ gives $\partial b_k/\partial s = c_{k-2}$, the same $c$
sequence shifted by one. So one $c$ pass supplies the whole 2 by 2 Jacobian

$$\begin{pmatrix}
\partial b_{n-1}/\partial r & \partial b_{n-1}/\partial s\\
\partial b_n/\partial r & \partial b_n/\partial s
\end{pmatrix}
= \begin{pmatrix} c_{n-2} & c_{n-3}\\ c_{n-1} & c_{n-2}\end{pmatrix},$$

and Bairstow is then Newton's method for the 2 by 2 system $b_{n-1}(r,s) = b_n(r,s) = 0$. That is
lesson 14's Newton for systems, at $n = 2$, applied to a problem set up by lesson 01's synthetic
division.

### 2.3 The Cauchy bound

Let $p(x) = a_0x^n + a_1x^{n-1} + \cdots + a_n$ with $a_0 \ne 0$, and set
$M = \max_{i\ge1}|a_i/a_0|$. Claim: every root satisfies $|r| \le 1 + M$.

*Proof.* Suppose $|r| > 1$ and $p(r) = 0$. Divide $p(r) = 0$ by $a_0$ and rearrange:

$$r^n = -\sum_{i=1}^{n}\frac{a_i}{a_0}r^{\,n-i}.$$

Take absolute values and use the triangle inequality and $|a_i/a_0| \le M$:

$$|r|^n \le M\sum_{i=1}^{n}|r|^{\,n-i} = M\big(|r|^{n-1} + \cdots + |r| + 1\big)
= M\,\frac{|r|^n - 1}{|r| - 1},$$

summing the geometric series, valid since $|r| > 1$. Drop the $-1$ in the numerator to weaken the
inequality:

$$|r|^n \le M\,\frac{|r|^n}{|r|-1} \implies |r| - 1 \le M \implies |r| \le 1 + M.$$

Roots with $|r| \le 1$ satisfy $|r| \le 1 + M$ trivially since $M \ge 0$. $\square$

The bound is often loose but it is free, and it is what makes Sturm-based isolation a **finite**
procedure: it turns "search the whole real line" into "search $[-1-M, 1+M]$".

### 2.4 One Graeffe step squares the roots

Write $p(x) = a_0\prod_{i=1}^n (x - r_i)$. Then

$$p(-x) = a_0\prod_i(-x - r_i) = a_0(-1)^n\prod_i(x + r_i).$$

Multiply:

$$p(x)\,p(-x) = a_0^2(-1)^n\prod_i (x-r_i)(x+r_i) = a_0^2(-1)^n\prod_i (x^2 - r_i^2).$$

Every term involves $x$ only through $x^2$, so define $P(y)$ by substituting $y = x^2$:

$$P(y) = (-1)^n p(x)p(-x)\big|_{x^2 = y} = a_0^2\prod_i(y - r_i^2).$$

$P$ has degree $n$ in $y$ and its roots are exactly $r_i^2$. $\square$

**The coefficients.** Writing out $p(x)p(-x)$ and collecting even powers gives the classical
formula

$$A_i = a_i^2 + 2\sum_{j\ge1}(-1)^j a_{i-j}a_{i+j},$$

with out-of-range terms taken as zero. After $k$ steps the roots are $r_i^{2^k}$, so their
relative sizes are exaggerated enormously and the roots can be read directly off the
coefficients, since $A_1/A_0 \approx -r_{\max}^{2^k}$.

**Why it is unusable.** The coefficients are being squared every step, so $\log|A|$ **doubles**
every step. Measured:

| polynomial | steps before overflow | $\log_{10}$ of the largest coefficient, per step |
|---|---:|---|
| degree 3, largest root 2 | 9 | 0.54, 0.72, 1.23, 2.41, 4.82, 9.63, 19.27 |
| degree 3, largest root 10 | 7 | 2.10, 4.19, 8.39, 16.78, 33.55, 67.10, 134.20 |
| degree 5, largest root 10 | 7 | 2.48, 4.51, 8.80, 17.55, 35.10, 70.20, 140.40 |
| degree 8, largest root 3 | 8 | 1.07, 1.46, 2.64, 5.23, 10.45, 20.90, 41.81 |
| degree 12, largest root 2 | 9 | 0.73, 0.76, 1.23, 2.41, 4.82, 9.63, 19.27 |

The doubling is exact after the first step or two, and double precision holds about $10^{308}$,
so $\log_{10}$ can double at most about 9 times from a starting value near 1. You get **7 to 9
steps**, and the count depends on the size of the largest root, not on the degree.

Seven steps means the roots have been raised to the power $2^7 = 128$, which does separate them,
but you then have to take a 128th root of a number known to only a few digits, and you have lost
all the sign information. Graeffe is a beautiful idea that floating point cannot carry.

### 3.1 Sturm isolation and where the accuracy goes

Measured (this is the experiment in section 3 of the lesson): with two roots at $1$ and $1+g$ and
a third at 5, Sturm counts correctly in $(0,3]$ down to $g = 10^{-4}$ and undercounts from
$g = 10^{-5}$ onwards. The companion matrix keeps three real roots well past $10^{-8}$.

**Where the accuracy is lost.** Instrument `sturm_sequence` and print the coefficient magnitudes
at each stage. The chain is built by repeated polynomial remainders, and the remainder step is a
subtraction of two nearly equal polynomials once the roots are close, so it is lesson 05's
cancellation applied repeatedly. Typically the **last two** members of the chain are where it
breaks: they are the ones whose leading coefficients involve the discriminant, which is
$O(g^2)$ for a root gap $g$, so at $g = 10^{-5}$ that is $10^{-10}$, and after several
cancelling divisions it is below the noise.

**With `fractions.Fraction`.** Build the same chain with exact rational arithmetic:

```python
from fractions import Fraction
coeffs = [Fraction(c) for c in exact_integer_coefficients]
```

Now Sturm resolves **any** gap, because the theorem is exact and so is the arithmetic. This
confirms the diagnosis: the theorem is fine, the chain construction in floating point is not.

The cost is that Fraction arithmetic has coefficient growth: the numerators and denominators in
the remainder chain roughly double in bit length at each stage, so a degree 20 chain can involve
integers with thousands of digits. That is why exact Sturm is used in computer algebra systems
and not in numerical libraries.

### 3.2 Bairstow on a degree 8 polynomial

Build $p$ with two real roots and three complex conjugate pairs:

```python
p = np.poly([1.0, -2.0, 0.5+1.5j, 0.5-1.5j, -1+0.5j, -1-0.5j, 2+0.3j, 2-0.3j])
p = np.real(p)                     # coefficients are real to roundoff
```

Run Bairstow, extract the quadratic factor, deflate by it, and repeat. Each pass reduces the
degree by 2, so four passes finish a degree 8 polynomial, leaving a constant.

Two things to check. First, that the recovered quadratic factors multiply back to $p$: compute
$\|\prod(\text{factors}) - p\|_\infty / \|p\|_\infty$ and confirm it is near $u$. Second, that
each quadratic's roots match the intended pair.

The practical difficulty is the starting guess $(r_0, s_0)$. Bairstow is Newton on a 2 by 2
system and inherits Newton's lack of global convergence. Common failures are convergence to a
factor you have already removed, and divergence to a quadratic with no roots near anything. The
standard remedies are to try several random starts, and to use the previous pass's converged
$(r,s)$ as the next pass's start when the roots are clustered.

### 3.3 Deflation with polishing

Polishing means taking one or two Newton steps on the **original** polynomial after each root is
found from the deflated one. It costs almost nothing and removes the error the deflated
polynomial introduced, since the correction is computed from clean data.

From the table in 1.3, polishing helps mostly in the **large-first** order, where it recovers an
order of magnitude at $n = 8$ ($6.6\times10^{-11}$ down to $6.0\times10^{-12}$). In the
small-first order the raw answers are already good, and polishing sometimes makes the maximum
error slightly worse, because a Newton step on an ill-conditioned root can move it in the wrong
direction when the residual is already at the noise level.

The honest conclusion: **polishing is a cheap safety net that helps most when you needed help.**
It does not rescue the method at degrees where deflation has already collapsed: at $n = 12$ both
orders fail to find all roots, and no amount of polishing fixes a root that was never found.

### 4.1 Deflation accuracy across degrees

Table given in exercise 1.3 above. The breaking points:

- **large-first deflation** breaks at $n = 10$,
- **small-first deflation** breaks at $n = 12$,
- **numpy.roots** (companion matrix) still returns all 14 roots at $n = 14$ with error
  $8.5\times10^{-7}$, and continues to degree 20 with error $9.3\times10^{-2}$ (lesson 12).

Note that even where deflation works it is not clearly better than the companion matrix, and it
is much more fragile. Its value is pedagogical, plus the specific case where you already know one
root exactly and want the rest.

### 4.2 Graeffe overflow

Table given in exercise 2.4. Key findings, all measured:

- **7 to 9 steps** before overflow in double precision.
- The count depends on the **largest root**, not the degree. Degree 3 and degree 12 both give 9
  steps when the largest root is 2; degree 3 and degree 5 both give 7 when it is 10.
- The logarithm of the largest coefficient **doubles exactly** each step after the first, as
  predicted, since the root magnitudes are being squared.

The reason degree does not matter is that overflow is driven by $|r_{\max}|^{2^k}$, and the
degree only affects the constant in front.

### 4.3 Bairstow against Muller for complex roots

Both find complex roots. They differ in almost every other respect.

| | Bairstow | Muller |
|---|---|---|
| arithmetic | real only | complex |
| finds | a conjugate **pair** at once | one root at a time |
| order | 2 (Newton on a 2 by 2 system) | about 1.84 |
| per step | two synthetic divisions, $O(n)$ | one evaluation plus a square root |
| applies to | polynomials only | any $f$ |
| starting guess | $(r_0, s_0)$, hard to choose | three points, easy |
| failure mode | diverges or returns a known factor | converges to a root you already have |

Across many random starting values, Muller is noticeably more **robust**, because a bad start
usually still converges to some root, whereas Bairstow's 2 by 2 Newton diverges more readily.
Bairstow is **faster per root found** when it works, because it gets two roots per convergence
and never touches complex arithmetic.

Practical verdict: Muller when you want reliability or when $f$ is not a polynomial; Bairstow
when you are writing a polynomial-specific routine in a language where complex arithmetic is
awkward. Neither is competitive with the companion matrix for general use.

### 5.1 Jenkins-Traub

The three stages, in brief:

1. **No-shift stage.** Apply the basic recurrence with shift zero, about 5 times, to emphasise
   the smallest root and damp the others.
2. **Fixed-shift stage.** Choose a shift $s$ on a circle of radius equal to the smallest root
   modulus (from a Cauchy-type bound) and iterate with that fixed shift until a convergence test
   on the sequence of $H$ polynomials fires.
3. **Variable-shift stage.** Update the shift at every step using the current estimate. This
   stage is the one that converges, and it does so **at least quadratically**, usually faster.

Stage 3 is the one to implement. The recurrence on the $H$ polynomials is

$$H^{(\lambda+1)}(x) = \frac{1}{x - s_\lambda}
\left(H^{(\lambda)}(x) - \frac{H^{(\lambda)}(s_\lambda)}{p(s_\lambda)}p(x)\right),$$

with the shift updated as $s_{\lambda+1} = s_\lambda - p(s_\lambda)/\bar H^{(\lambda+1)}(s_\lambda)$,
where the bar denotes the normalised $H$.

**Against the companion matrix on hard polynomials.** Jenkins-Traub is generally more accurate on
polynomials with widely spread roots, because it works with the coefficients directly and
implicitly deflates in a numerically careful order, and it is $O(n^2)$ per root rather than the
companion matrix's $O(n^3)$ overall. It is what MATLAB's `roots` used historically and what many
Fortran libraries still use. On Wilkinson-type polynomials it does **not** beat the companion
matrix by much, for the reason lesson 12 exercise 5.2 gives: the conditioning is in the problem,
and no algorithm escapes it.

### 5.2 Colleague and comrade matrices

A **colleague matrix** does for the Chebyshev basis what the companion matrix does for the
monomial basis. Write $p = \sum_k c_kT_k(x)$; then the colleague matrix is tridiagonal plus a
rank-one correction in the last row, and its eigenvalues are the roots of $p$. **Comrade
matrices** generalise this to any basis satisfying a three-term recurrence, which is every family
of orthogonal polynomials (lesson 46).

**The measurement to make.** Take the Wilkinson polynomial, convert it to a Chebyshev basis
scaled to $[0, 21]$, build the colleague matrix, and compute its eigenvalue condition numbers.
Compare with the companion matrix's.

**What you find.** The colleague matrix's eigenvalue conditioning is **dramatically** better,
often by many orders of magnitude, for polynomials whose roots lie in the interval the Chebyshev
basis is scaled to. The reason is the same as lesson 12 exercise 4.3: the Chebyshev coefficients
of a polynomial with roots in $[-1,1]$ do not span many orders of magnitude, while the monomial
coefficients do.

**The connection to lesson 12.** Lesson 12's point was that a badly conditioned problem cannot be
rescued by a better algorithm. This is the other half of that statement: it **can** be rescued by
a better **representation**, because a different representation is a different problem, with
different data and different conditioning. The roots are the same; the numbers you use to
describe them are not.

This is why Chebfun represents functions in a Chebyshev basis and finds roots via colleague
matrices, and why it can reliably find the roots of a degree 1000 polynomial, which is
unthinkable in the monomial basis.

### 5.3 What "its roots" means for inexact coefficients

**The definition.** Given $p$ with coefficients known to relative accuracy $\epsilon$, the honest
answer is the **pseudozero set** (lesson 12 exercise 5.1):

$$Z_\epsilon(p) = \{z : |p(z)| \le \epsilon\textstyle\sum_j |a_j|\,|z|^j\}.$$

Compute it by evaluating $|p(z)|$ on a grid over $\mathbb{C}$, dividing by
$\sum_j|a_j||z|^j$, and contouring at $\epsilon$.

**For a degree 10 polynomial known to 3 digits**, meaning $\epsilon = 10^{-3}$, the set is
typically several large blobs rather than 10 points. Roots that are well separated relative to
their sensitivity give small isolated blobs. Roots that are close, or large, merge into single
connected regions that often extend well off the real axis.

**What should be reported to a user.** Four rules, in order of importance:

1. **Report connected components, not points.** If two nominal roots lie in one component of
   $Z_\epsilon$, there is no fact of the matter about which is which, or even whether there are
   two real roots or one complex pair. Report the component.
2. **Report a radius with every isolated root.** For a component containing one root, report the
   centre and the radius. The radius is the answer to the question the user is actually asking.
3. **Never print more digits than $\epsilon$ supports.** Printing `2.7182818284` for a root
   determined to $\pm 0.01$ is a lie told by a print statement, and it is the single most common
   way numerical software misleads people.
4. **Say when the components merge.** "These 4 nominal roots are indistinguishable at your input
   accuracy" is far more useful than 4 confident and wrong numbers.

**The general principle.** This is the same conclusion as lesson 12 exercise 5.1 and lesson 06:
the output of an ill-conditioned computation is a **set**, and the size of that set is determined
by the input accuracy, not by the algorithm. Software that reports a point when the truth is a
blob has thrown away the only information the user needed.

---

## Lesson 14, Nonlinear Systems of Equations

### 1.1 Why there is no bisection for systems

**The intermediate value theorem has no $n$-dimensional analogue that gives you a root.**

In one dimension, $f(a) < 0 < f(b)$ forces a crossing, because to get from negative to positive a
continuous function must pass through zero, and there is only one way to travel from $a$ to $b$.

In $n$ dimensions the natural generalisation would be: if $\mathbf{F}$ has "opposite signs" on
the boundary of a box, there is a root inside. But **"opposite signs" does not mean anything** for
a vector. $\mathbf{F}$ has $n$ components, and each can have any sign at each of the box's
corners. There is no sign pattern that forces a root.

Concretely, a continuous map can be nonzero on the entire boundary of a box and vanish inside, or
be nonzero on the boundary and never vanish at all, and no amount of boundary data distinguishes
the two. Consider $\mathbf{F}(x,y) = (x^2+y^2-\tfrac14,\ 0)$ on the unit square: the second
component is zero everywhere, the first changes sign, and there is a whole circle of roots. Now
change the first component's constant and the roots vanish while the boundary values barely move.

**What does survive** is the **topological degree**, computed from the boundary behaviour, which
does guarantee a root when nonzero. But it is expensive to compute, it can be zero when roots
exist (two roots of opposite index cancel), and it does not tell you where the root is. There are
generalised bisection algorithms built on it, and they are used only when nothing else works.

The bare practical statement: **every method in lesson 14 is an open method, and every one of
them can fail.** That is the price of $n > 1$.

### 1.2 Why Broyden is not always better

Broyden avoids the Jacobian, but it pays in three ways.

**It converges superlinearly, not quadratically.** Near the root Newton roughly doubles the
number of correct digits per step, while Broyden does not. That means more iterations, and each
iteration still costs a linear solve.

**The approximation can go stale or wrong.** $B_k$ is built from the steps actually taken, so it
is accurate only in the directions already explored. In $n$ dimensions, $k$ steps give information
about a $k$-dimensional subspace, and the other $n - k$ directions are still whatever $B_0$ said.
For large $n$ and few steps, most of $B$ is a guess.

**It is less robust from a bad start.** Newton's Jacobian is correct at the current point, so its
direction is at least locally right. Broyden's may not be, and the rank-one update can make $B$
singular or nearly so, at which point the method fails outright.

**When Newton wins:** when the Jacobian is analytically available and cheap, when $n$ is small,
when the start is poor, or when you need the fewest possible iterations because each function
evaluation is enormously expensive.

**When Broyden wins:** when $\mathbf{F}$ is expensive and $J$ has to be built by finite
differences, which costs $2n$ extra evaluations per step. Measured on a family of test systems
with a finite-difference Jacobian, total $\mathbf{F}$ evaluations to $10^{-12}$:

| $n$ | Broyden | Newton with FD Jacobian |
|---:|---:|---:|
| 2 | 10 | 16 |
| 5 | 17 | 34 |
| 12 | 30 | 76 |
| 20 | 46 | 124 |
| 40 | 86 | 244 |
| 80 | 165 | 323 |

Broyden wins at **every** size here, and by a growing margin, because Newton's cost per step
grows like $2n$ while Broyden's stays at 1. The crossover exercise 4.2 asks about therefore does
not exist in this comparison: it exists only against Newton with an **analytic** Jacobian, where
Newton's per-step evaluation count is 1 and its better rate wins.

### 1.3 $\|G'\|_\infty = 1.4$ but $\rho(G') = 0.8$

**Yes, it converges.** The spectral radius is what Theorem 14.1 requires, and $0.8 < 1$.

The norm being above 1 only means the error can **grow for a few steps** before it starts
shrinking. Since $\mathbf{e}_k \approx (G')^k\mathbf{e}_0$ and

$$\|(G')^k\|^{1/k} \to \rho(G')$$

by Gelfand's formula, the eventual decay rate is 0.8 per step whatever the norm says. Early
steps can be governed by the norm, so you might see the error rise from 1 to 1.4 to 1.96 before
turning around. This is **transient growth**, and it is the finite-dimensional version of the
non-normality phenomenon that lesson 40 develops.

Two practical warnings that follow:

- A convergence test based on $\|\mathbf{x}_{k+1} - \mathbf{x}_k\|$ increasing does **not** mean
  divergence. Do not give up early on a non-normal iteration.
- Transient growth can take the iterate outside the region where the linearisation is valid,
  and then the nonlinear problem does something the linear theory did not predict. So the answer
  "yes it converges" is a statement about local behaviour, and a large $\|G'\|$ with a small
  $\rho(G')$ means the basin of attraction may be small.

### 2.1 Proving Theorem 14.1

**The subtlety first.** The mean value theorem in the form $\mathbf{G}(x) - \mathbf{G}(y) =
G'(\xi)(x-y)$ is **false** in $n$ dimensions for $n > 1$. Each component needs its own
intermediate point, and there is generally no single $\xi$ that works for all of them.

The correct tool is the **integral form**:

$$\mathbf{G}(\mathbf{x}) - \mathbf{G}(\mathbf{y})
= \left(\int_0^1 G'\big(\mathbf{y} + t(\mathbf{x}-\mathbf{y})\big)\,dt\right)(\mathbf{x}-\mathbf{y}),$$

obtained by applying the fundamental theorem of calculus to $t \mapsto \mathbf{G}(\mathbf{y} +
t(\mathbf{x}-\mathbf{y}))$. This is exact, and the matrix in brackets is an average of Jacobians
rather than a Jacobian at a point.

**The proof.** Let $\mathbf{r}$ be the fixed point, $\mathbf{e}_k = \mathbf{x}_k - \mathbf{r}$.
Apply the integral form with $\mathbf{x} = \mathbf{x}_k$ and $\mathbf{y} = \mathbf{r}$:

$$\mathbf{e}_{k+1} = \mathbf{G}(\mathbf{x}_k) - \mathbf{G}(\mathbf{r}) = A_k\,\mathbf{e}_k,
\qquad A_k = \int_0^1 G'(\mathbf{r} + t\mathbf{e}_k)\,dt.$$

Since $\rho(G'(\mathbf{r})) < 1$, pick $\epsilon > 0$ with $\rho + 2\epsilon < 1$. A standard fact
about matrices says there is a norm $\|\cdot\|_\ast$ with $\|G'(\mathbf{r})\|_\ast \le \rho +
\epsilon$. (This is the fact that does the real work, and it is why the spectral radius and not a
norm is the right condition: you get to **choose** the norm.)

By continuity of $G'$, there is a $\delta$ with $\|G'(\mathbf{z}) - G'(\mathbf{r})\|_\ast \le
\epsilon$ whenever $\|\mathbf{z}-\mathbf{r}\|_\ast \le \delta$. For $\|\mathbf{e}_k\|_\ast \le
\delta$ the whole segment stays inside the ball, so

$$\|A_k\|_\ast \le \int_0^1\|G'(\mathbf{r}+t\mathbf{e}_k)\|_\ast dt
\le \rho + 2\epsilon < 1.$$

Hence $\|\mathbf{e}_{k+1}\|_\ast \le (\rho+2\epsilon)\|\mathbf{e}_k\|_\ast$, the iterate stays in
the ball, and the argument repeats. So $\mathbf{e}_k \to 0$ linearly, and letting $\epsilon \to
0$ gives the asymptotic rate $\rho(G'(\mathbf{r}))$. $\square$

Measured in lesson 14 section 2: the observed decay rate matches $\rho(G')$ to within $10^{-3}$.

### 2.2 Broyden's update is the unique least-change solution

**Problem.** Minimise $\|B_{k+1} - B_k\|_F$ subject to $B_{k+1}\mathbf{s} = \mathbf{y}$.

Write $B_{k+1} = B_k + \Delta$. The constraint becomes $\Delta\mathbf{s} = \mathbf{y} -
B_k\mathbf{s} =: \mathbf{d}$, and we minimise $\|\Delta\|_F$.

**Solution.** Decompose any candidate $\Delta$ using the orthogonal projector onto
$\mathrm{span}(\mathbf{s})$, namely $P = \mathbf{s}\mathbf{s}^T/(\mathbf{s}^T\mathbf{s})$:

$$\Delta = \Delta P + \Delta(I - P).$$

The two pieces have orthogonal ranges in the Frobenius inner product, since
$\langle \Delta P, \Delta(I-P)\rangle_F = \mathrm{tr}(P^T\Delta^T\Delta(I-P)) = 0$ using
$P^T = P = P^2$ and $P(I-P) = 0$. So by Pythagoras

$$\|\Delta\|_F^2 = \|\Delta P\|_F^2 + \|\Delta(I-P)\|_F^2.$$

The constraint determines the first piece completely:

$$\Delta P = \Delta\frac{\mathbf{s}\mathbf{s}^T}{\mathbf{s}^T\mathbf{s}}
= \frac{(\Delta\mathbf{s})\mathbf{s}^T}{\mathbf{s}^T\mathbf{s}}
= \frac{\mathbf{d}\,\mathbf{s}^T}{\mathbf{s}^T\mathbf{s}},$$

which does not depend on $\Delta$ at all beyond the constraint. The constraint says nothing about
the second piece, so the minimum is attained exactly when $\Delta(I-P) = 0$, and it is the
**unique** minimiser because the second term is a strictly positive contribution otherwise.

Therefore

$$\Delta = \frac{\mathbf{d}\,\mathbf{s}^T}{\mathbf{s}^T\mathbf{s}}
= \frac{(\mathbf{y} - B_k\mathbf{s})\mathbf{s}^T}{\mathbf{s}^T\mathbf{s}},$$

which is Broyden's update, and it is unique. $\square$

The tests in `tests/test_nlsystems.py` verify both halves numerically: that the update satisfies
$B_{k+1}\mathbf{s} = \mathbf{y}$ exactly, and that every other matrix satisfying that constraint
(constructed as $B_{k+1} + M$ with $M\mathbf{s} = 0$) is at least as far from $B_k$ in Frobenius
norm.

**Why "least change" is the right principle.** Every step gives $n$ pieces of information about
an $n^2$-entry matrix, so the problem is massively underdetermined and something must be assumed.
Assuming "change as little as possible" says: keep everything you have learned so far, and add
only what the new data forces. That is the same principle behind BFGS, and it is why the
quasi-Newton family exists at all.

### 2.3 Newton for systems is quadratic

Let $\mathbf{r}$ satisfy $\mathbf{F}(\mathbf{r}) = 0$ with $J(\mathbf{r})$ nonsingular and $J$
Lipschitz with constant $L$ near $\mathbf{r}$.

Taylor with integral remainder, at $\mathbf{x}_k$ evaluated at $\mathbf{r}$:

$$0 = \mathbf{F}(\mathbf{r}) = \mathbf{F}(\mathbf{x}_k) + J(\mathbf{x}_k)(\mathbf{r}-\mathbf{x}_k)
+ \mathbf{R}_k,$$

where

$$\mathbf{R}_k = \int_0^1\big[J(\mathbf{x}_k + t(\mathbf{r}-\mathbf{x}_k)) - J(\mathbf{x}_k)\big]
(\mathbf{r}-\mathbf{x}_k)\,dt,$$

so by the Lipschitz bound $\|\mathbf{R}_k\| \le \tfrac{L}{2}\|\mathbf{e}_k\|^2$ with
$\mathbf{e}_k = \mathbf{x}_k - \mathbf{r}$.

The Newton step satisfies $J(\mathbf{x}_k)(\mathbf{x}_{k+1}-\mathbf{x}_k) = -\mathbf{F}(\mathbf{x}_k)$.
Subtract that from the Taylor identity rearranged:

$$J(\mathbf{x}_k)(\mathbf{x}_{k+1} - \mathbf{r}) = \mathbf{R}_k,$$

hence

$$\|\mathbf{e}_{k+1}\| \le \|J(\mathbf{x}_k)^{-1}\|\,\frac{L}{2}\|\mathbf{e}_k\|^2. \qquad \square$$

Since $J(\mathbf{r})$ is nonsingular, $\|J(\mathbf{x}_k)^{-1}\| \le 2\|J(\mathbf{r})^{-1}\| =:
2\beta$ for $\mathbf{x}_k$ close enough, giving $\|\mathbf{e}_{k+1}\| \le \beta L\|\mathbf{e}_k\|^2$,
which is quadratic convergence.

**Compared with Theorem 11.1**, the only changes are: $|f'(r)| \ne 0$ becomes "$J(\mathbf{r})$ is
nonsingular", division by $f'$ becomes multiplication by $J^{-1}$ (in practice a linear solve),
$|f''|$ bounded becomes $J$ Lipschitz, and the scalar mean value theorem becomes its integral
form for the reason in exercise 2.1. The structure of the proof is identical.

### 2.4 The $O(n^2)$ Broyden variant

**Sherman-Morrison.** For invertible $A$ and vectors $\mathbf{u},\mathbf{v}$ with
$1 + \mathbf{v}^TA^{-1}\mathbf{u} \ne 0$,

$$(A + \mathbf{u}\mathbf{v}^T)^{-1} = A^{-1}
- \frac{A^{-1}\mathbf{u}\mathbf{v}^TA^{-1}}{1 + \mathbf{v}^TA^{-1}\mathbf{u}}.$$

Apply it with $A = B_k$, $\mathbf{u} = (\mathbf{y}-B_k\mathbf{s})/(\mathbf{s}^T\mathbf{s})$ and
$\mathbf{v} = \mathbf{s}$. Writing $H_k = B_k^{-1}$ and simplifying, the denominator becomes
$\mathbf{s}^TH_k\mathbf{y}/(\mathbf{s}^T\mathbf{s})$ and the whole update collapses to the
elegant form

$$H_{k+1} = H_k + \frac{(\mathbf{s} - H_k\mathbf{y})\,\mathbf{s}^TH_k}{\mathbf{s}^TH_k\mathbf{y}}.$$

The step is then $\mathbf{s} = -H_k\mathbf{F}(\mathbf{x}_k)$, a matrix-vector product.

**Cost.** Per step: one matrix-vector product for the step ($n^2$ flops), one for
$H_k\mathbf{y}$ ($n^2$), one outer product update ($n^2$). Total $O(n^2)$, against
$O(n^3)$ for the version that solves $B_k\mathbf{s} = -\mathbf{F}$ from scratch.

**What is given up.** Three things, and the third is the one that matters.

1. **You cannot monitor conditioning.** With $B$ and a factorisation you can watch for
   near-singularity. With $H$ you only find out when the steps go wild.
2. **You cannot easily impose structure.** If $J$ is sparse or banded, $B$ can be kept that way
   and $H$ cannot, since the inverse of a sparse matrix is dense.
3. **Numerical stability is worse.** Explicitly propagating an inverse accumulates error in a way
   that solving a fresh linear system does not, and errors in $H$ are never corrected. This is the
   same reason lesson 20 says never to compute $A^{-1}$ when you want to solve $A\mathbf{x} =
   \mathbf{b}$.

The practical compromise, used in real codes, is to keep a **factorisation** of $B_k$ and update
the factorisation under the rank-one change, which also costs $O(n^2)$ but retains stability and
the ability to monitor conditioning.

### 3.1 The Stewart platform

The problem: a triangular platform is held by three legs of known lengths $p_1, p_2, p_3$
attached to known points on a fixed base. Find the platform's pose, meaning two coordinates
$(x, y)$ and a rotation angle $\theta$.

Each leg gives one equation "the distance from this base point to this platform point equals
$p_i$", which is quadratic in $x, y$ and involves $\cos\theta, \sin\theta$. Three legs give three
equations in three unknowns. Sauer's formulation eliminates $x$ and $y$ to reduce it to a single
equation in $\theta$, which turns out to have up to **six** solutions.

Two things worth doing beyond just solving it.

**Find all the solutions, not one.** Newton finds whichever one it drifts to. Sample $\theta_0$
across $[0, 2\pi)$ on a fine grid, run Newton from each, and collect the distinct answers. This
is where lesson 09's bracketing on the reduced one-dimensional equation is genuinely better than
Newton, because it can prove it found them all.

**Interpret the multiplicity physically.** Multiple solutions mean the leg lengths do not
determine the pose. That is a real engineering problem, not a numerical artefact, and it is why
Stewart platforms are instrumented with extra sensors.

### 3.2 The $O(n^2)$ Broyden in practice

Implement the $H$ update from exercise 2.4 and check that the iterates match the $B$ version to
roundoff. They should agree to about $10^{-12}$ relative for the first several steps, then drift
apart slowly as the two accumulate different roundoff, and converge to the same root.

Do not expect exact agreement. Sherman-Morrison is mathematically an identity and numerically is
not, and the drift you observe is a direct measurement of the stability cost from exercise 2.4.
Plot $\|\mathbf{x}_k^{(H)} - \mathbf{x}_k^{(B)}\|$ against $k$ to see it.

The timing crossover is around $n = 50$ to 100 on typical hardware, below which the $O(n^3)$
solve is not the bottleneck anyway.

### 3.3 Newton on a discretised boundary value problem

Discretise $-u'' + f(u) = g$ on $[0,1]$ with $u(0) = u(1) = 0$ using $n = 200$ interior points and
the standard second difference. The result is

$$\mathbf{F}(\mathbf{u}) = A\mathbf{u} + f(\mathbf{u}) - \mathbf{g} = 0,$$

where $A$ is the tridiagonal second difference matrix scaled by $1/h^2$.

The Jacobian is $J = A + \mathrm{diag}(f'(u_i))$, which is **tridiagonal**. That is the whole
point of the exercise.

Timings to expect: the dense solve is $O(n^3) = 8\times10^6$ flops per step, while a tridiagonal
solve is $O(n) = 200$. At $n = 200$ that is a factor of several hundred, and it grows without
bound. `scipy.sparse.linalg.spsolve` on a `scipy.sparse` matrix, or better
`scipy.linalg.solve_banded`, exploits it.

The two lessons: **the structure of the Jacobian is inherited from the discretisation**, and
Newton's cost is entirely determined by the linear solve, which is why Part 3 exists and why
lesson 21 (banded and sparse direct methods) is where this gets done properly.

### 4.1 Basins of attraction for the test system

Colour a grid of starting points by which of the two solutions Newton reaches, with a third
colour for divergence.

The picture has the same character as the one-dimensional Newton fractal (lesson 11 exercise
3.2): two large basins around the solutions, an intricate boundary between them, and thin
filaments of one basin threading deep into the other. The filaments emanate from the curve where
$\det J = 0$, since near there the Newton step is enormous and can land anywhere.

**Plain against damped.** The damped map's basins are much larger and much smoother, and the
divergence region is nearly eliminated. This is the visual version of the table in section 4.

But look carefully at the **boundary** in the damped picture, and compare iteration counts, not
just outcomes. Damping converts many divergences into slow convergences, so the basins grow while
the average iteration count in the newly captured regions is high. That is consistent with the
Rosenbrock result in section 4: damping trades speed for reliability, and where it fails to be
faster it is often much slower.

### 4.2 The Broyden crossover

Measured in exercise 1.2 above. Against Newton with a **finite-difference** Jacobian, Broyden
wins at every $n$ from 2 to 80, and the margin widens, because Newton's per-step cost grows like
$2n$ while Broyden's is constant.

To find a real crossover you have to change the comparison. Against Newton with an **analytic**
Jacobian costing one evaluation-equivalent per step, Newton's quadratic rate wins at every $n$
in the same family, since both cost $O(n^3)$ per step for the solve and Newton takes fewer steps.

So the honest answer to "where is the crossover" is: **it is not a crossover in $n$, it is a
crossover in how the Jacobian is obtained.** Finite differences means Broyden; analytic or
automatic differentiation means Newton.

The one place $n$ genuinely matters is memory: Broyden stores a dense $B$ or $H$, so it is
$O(n^2)$ storage and unusable for very large $n$. That is what limited-memory methods (L-BFGS,
lesson 86) and Newton-Krylov (exercise 5.1) exist for.

### 4.3 Bad scaling

Rescale so $x' = 10^6 x$. Then $\mathbf{F}$ becomes badly scaled, the Jacobian's columns differ
by $10^6$, and $\kappa(J)$ rises by about $10^6$.

**What changes and what does not.** Newton's iteration is **invariant under linear rescaling of
the variables**, in exact arithmetic: if $\mathbf{x}' = S\mathbf{x}$ then the Newton step
transforms consistently and the iterates correspond exactly. So in exact arithmetic the iteration
count is unchanged.

In floating point it is a different story. The linear solve $J\mathbf{s} = -\mathbf{F}$ is now
being done with an ill-conditioned $J$, so $\mathbf{s}$ picks up an error of relative size
$\kappa(J)u \approx 10^{-10}$, which limits the accuracy of the step and therefore the accuracy
of the final answer. And the **convergence test** is no longer scale-free: $\|\mathbf{s}\| \le
\text{tol}$ means something completely different for the two variables.

So you should measure: $\kappa(J)$ up by about $10^6$, iteration count roughly unchanged or
slightly up, and final accuracy down by about $10^6$.

**Rescaling properly** means choosing $S$ so the columns of $JS$ have similar norms, which is
column equilibration (lesson 19). After that, $\kappa$ returns to its original value and the
accuracy comes back. The standard practical rule is to scale variables so they are all $O(1)$ in
the units of the problem, and to scale equations so their residuals are comparable. Doing both is
**two-sided** scaling, and it is one of the highest-value cheap things you can do to a nonlinear
solver.

### 5.1 Newton-Krylov

The idea: never form $J$. Solve $J\mathbf{s} = -\mathbf{F}$ with GMRES (lesson 27), which needs
only products $J\mathbf{v}$, and approximate those by a directional difference

$$J\mathbf{v} \approx \frac{\mathbf{F}(\mathbf{x}+\epsilon\mathbf{v}) - \mathbf{F}(\mathbf{x})}{\epsilon},
\qquad \epsilon \approx \frac{\sqrt{u}\,(1 + \|\mathbf{x}\|)}{\|\mathbf{v}\|}.$$

One extra $\mathbf{F}$ evaluation per Krylov iteration, no Jacobian, no $O(n^3)$ solve, and
$O(n)$ storage per Krylov vector.

**The three things to get right.**

1. **The choice of $\epsilon$.** Too small and the difference is roundoff; too large and the
   product is inaccurate. The $\sqrt{u}$ balance is lesson 60's forward difference analysis
   applied here, and the scaling by $\|\mathbf{v}\|$ matters because $\mathbf{v}$ is a Krylov
   vector of arbitrary size.
2. **Inexact Newton.** You do not need to solve the linear system accurately, especially early
   on. The Eisenstat-Walker forcing terms set the GMRES tolerance adaptively as
   $\|\mathbf{F}\|$ shrinks, so early steps are cheap and later steps are accurate. Getting this
   right is most of the practical speedup.
3. **Preconditioning.** GMRES on an unpreconditioned discretised PDE converges slowly, and the
   whole method is only as good as the preconditioner. This is where domain knowledge enters.

Test it on the discretised nonlinear Poisson problem from exercise 3.3, scaled up to $n = 10^5$,
where forming $J$ densely is impossible and even a sparse direct solve is expensive in 3D.
`scipy.optimize.root(method="krylov")` implements all of this and is a good reference to check
against.

### 5.2 Continuation

Build a homotopy $\mathbf{H}(\mathbf{x}, t)$ with $\mathbf{H}(\cdot, 0)$ easy and
$\mathbf{H}(\cdot, 1) = \mathbf{F}$. The standard choices:

$$\mathbf{H}(\mathbf{x},t) = t\mathbf{F}(\mathbf{x}) + (1-t)(\mathbf{x} - \mathbf{x}_0)
\quad\text{(Newton homotopy, trivially solvable at } t = 0),$$

$$\mathbf{H}(\mathbf{x},t) = \mathbf{F}(\mathbf{x}) - (1-t)\mathbf{F}(\mathbf{x}_0)
\quad\text{(residual homotopy)}.$$

March $t$ from 0 to 1 in steps, solving with Newton at each $t$ and using the previous solution
as the start. Since consecutive problems are close, Newton always starts near its root, which is
exactly the situation where it is reliable.

**The failure mode, and the fix.** The solution path can **turn back** in $t$, at a fold point
where $\partial\mathbf{H}/\partial\mathbf{x}$ is singular. Then no amount of stepping forward in
$t$ works. The fix is **pseudo-arclength continuation**: parametrise the path by arclength
$s$ rather than by $t$, treat $t$ as one more unknown, and add the normalisation equation

$$\dot{\mathbf{x}}^T(\mathbf{x} - \mathbf{x}_{\text{prev}})
+ \dot t\,(t - t_{\text{prev}}) - \Delta s = 0.$$

The system is now $n+1$ equations in $n+1$ unknowns and is nonsingular at folds, so the path can
be traced around them.

A good test problem is a system with a known fold, such as the buckling of an elastic beam or the
bratu problem $-u'' = \lambda e^u$, where the solution branch turns back at a critical $\lambda$
and Newton from a fresh start fails on the upper branch for every guess you try.

### 5.3 The quasi-Newton family

**The shared structure.** Every quasi-Newton method updates an approximate Jacobian or Hessian by
a low-rank change subject to a secant condition, and every one of them can have its inverse
updated by **Sherman-Morrison-Woodbury**:

$$(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}.$$

Rank one ($U, V$ vectors) gives Broyden and SR1. Rank two ($U, V$ with two columns) gives DFP and
BFGS. In every case the inverse update costs $O(n^2)$ instead of $O(n^3)$.

**Why symmetric problems get BFGS instead.** Three reasons, and they compound.

1. **The Hessian is symmetric.** When $\mathbf{F} = \nabla f$ for a scalar $f$, the Jacobian is
   the Hessian and is symmetric by equality of mixed partials. Broyden's update does **not**
   preserve symmetry, so it throws away known structure and wastes half the storage and half the
   information.
2. **The Hessian at a minimum is positive definite**, and BFGS's update **preserves positive
   definiteness** whenever the curvature condition $\mathbf{s}^T\mathbf{y} > 0$ holds. That
   guarantees the step $-H\nabla f$ is a **descent direction**, which is what makes the line
   search work. Broyden gives no such guarantee, and a non-descent direction breaks the whole
   optimization framework.
3. **Rank two is needed.** A symmetric rank-one update (SR1) can satisfy the secant condition and
   preserve symmetry, but it can have a vanishing denominator and can destroy positive
   definiteness. Rank two is the smallest change that satisfies the secant condition **and**
   preserves both symmetry and definiteness.

BFGS is derived by the same least-change principle as Broyden (exercise 2.2), but minimising a
weighted Frobenius norm over the set of **symmetric** matrices satisfying the secant condition.
Same principle, larger constraint set, different answer. Lesson 86 does this properly.

---

## Where these solutions sit in the course

Part 2 established the machinery that Part 3 onwards assumes:

- **Convergence order and the efficiency index**, used to compare every iterative method in the
  course, including the Krylov methods of Part 4 and the optimization methods of Part 12.
- **Conditioning against stability**, met here for roots and coefficients, met again in Part 3
  for linear systems, in Part 5 for eigenvalues, and in Part 7 for interpolation.
- **Newton's method**, which reappears as the inner solver for implicit ODE methods (Part 9),
  for nonlinear least squares (lesson 33), and for constrained optimization (Part 12).
- **The observation that division becomes a linear solve**, which is why Part 3 is the longest
  part of the course.
