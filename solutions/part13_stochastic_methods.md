# Solutions: Part 13, Stochastic and Monte Carlo Methods

Solutions to every exercise in lessons 91 to 93, 63 in all: 21 for lesson 91, 21 for lesson 92, 21 for lesson 93.

Every number quoted here was measured by running the code, on this repository, with the seeds
shown. Where a measurement contradicted the claim the exercise expects, the measurement is
reported and the claim is corrected. Several exercises in this part end in a negative result for
that reason, and those are marked rather than replaced with an example that would have worked.

Run any block from the repository root. Each one is self contained apart from `nalib`, and the
blocks within one lesson share a namespace, so they are meant to be run in order.

```python
import sys
sys.path.insert(0, "src")
```

---

## Lesson 91, Random Number Generation

### 1.1 The recurrence and what bounds its period

$$
x_{k+1} = (a\,x_k + c) \bmod m ,
$$

with the output $x_k/m$. The multiplier $a$, the increment $c$ and the modulus $m$ define it
completely, and $x_0$ is the seed.

**The period is bounded by $m$**, and the reason is that the state is a single integer in
$\{0,\dots,m-1\}$, so there are only $m$ possible states. After at most $m+1$ steps some state must
repeat, and because $x_{k+1}$ depends only on $x_k$ the sequence from that point is identical to the
sequence from the first occurrence. So the generator enters a cycle and stays in it.

Two consequences worth separating. The bound is on the **state**, not on the output: a generator with
a larger state can have a much longer period while producing the same sized numbers, which is what
the Mersenne Twister does with its $19937$ bit state. And the period is a property of $(a, c, m)$ and
the seed together; a badly chosen seed can land on a short cycle even when a full period cycle
exists.

### 1.2 The Hull-Dobell conditions

The period is $m$ for every seed exactly when all three hold:

- $c$ and $m$ are coprime,
- $a - 1$ is divisible by every prime factor of $m$,
- $a - 1$ is divisible by $4$ if $m$ is.

**Why each is needed.**

*Coprimality of $c$ and $m$.* If a prime $p$ divides both, then $x_{k+1} \equiv a x_k \pmod p$, so the
sequence taken mod $p$ never leaves the multiplicative structure and cannot visit every residue. With
$c = 0$ the value $0$ is a fixed point, which is the extreme case.

*Divisibility of $a-1$ by every prime factor.* Writing $a = 1 + d$, the $k$-th iterate is
$x_k = x_0 + k c + d(\ldots)$, and the increment part advances by $c$ each step. If some prime $p$
divides $m$ but not $d$, the orbit closes early modulo $p^{e}$ and the period divides $m/p$.

*The factor of $4$.* A separate condition because $2$ is the only prime for which the multiplicative
group mod $2^{e}$ is not cyclic. Without it the period modulo $4$ is $2$ rather than $4$, and the
whole sequence inherits that.

Exercise 2.1 proves sufficiency, and the lesson's sweep confirms the equivalence over all $4096$
combinations at $m = 64$.

### 1.3 Why the tuples lie on hyperplanes

With $c = 0$ the recurrence gives $x_{k+j} \equiv a^{j}x_k \pmod m$, so the point
$(x_k, x_{k+1}, \dots, x_{k+d-1})$ is determined by $x_k$ alone. As $x_k$ runs over its orbit, those
points trace out the set

$$
L = \big\{\,(t,\; a t,\; a^{2}t,\; \dots) \bmod m \;:\; t \in \mathbb{Z}\,\big\} ,
$$

which is a **lattice**: it is closed under addition and under integer multiples, because the whole
construction is linear.

Every lattice in $\mathbb{R}^{d}$ lies on a family of parallel hyperplanes, one family for each
nonzero vector $v$ of the **dual** lattice: the points satisfy $v \cdot x \in \mathbb{Z}$, so they lie
on the planes $v\cdot x = 0, 1, 2, \ldots$ The spacing between consecutive planes is $1/\lVert
v\rVert$, so a **short** dual vector means widely spaced planes and a visibly gappy point set.

That is exactly what the spectral test measures, and exercise 3.1 computes it by lattice reduction.

A general LCG with $c \ne 0$ has the same structure applied to the differences $x_{k+1} - x_k$, so
the conclusion is unchanged.

### 1.4 Why `rand() % n` is bad, in two ways

**The bottom bits have almost no period.** With a power of two modulus, arithmetic mod $2^{j+1}$
closes on itself, so bit $j$ depends only on the bottom $j+1$ bits of the state and has period at
most $2^{j+1}$. A remainder by $8$ reads the bottom three bits, whose period is $8$, so
`rand() % 8` cycles through eight values forever whatever the generator's own period is. The
lesson measures periods of exactly $2, 4, 8, 16, 32, 64$ for the bottom six bits.

**The remainder is biased unless $n$ divides the range.** If the generator produces integers
uniformly on $\{0,\dots,M-1\}$ and $n$ does not divide $M$, then the values $0,\dots,(M \bmod n)-1$
occur one more time than the rest. The bias is of relative size $n/M$, which is negligible for small
$n$ and large $M$ and is not negligible when $n$ is a substantial fraction of $M$.

The two are independent faults. The second afflicts even a perfect generator; the first afflicts even
a perfectly unbiased reduction.

```python
import numpy as np
from nalib import rng as rg

out = rg.the_low_bits_are_worse()
print("the first fault: the bottom bits repeat")
print(f"{'bit':>5}{'predicted period':>19}{'measured':>11}")
for row in out["rows"]:
    print(f"{row['bit']:>5}{row['predicted_period']:>19}{row['measured_period']:>11}")

print("\na die rolled by taking a remainder, from the bottom three bits:")
values = rg.lcg(48, *rg.FAMOUS["ansi c"], seed=1)
print(f"  {(values % 8)}")
print(f"  the pattern repeats with period {out['rows'][2]['measured_period']}")

print("\nthe second fault: the remainder is biased when n does not divide the range")
span, n = 10, 3
counts = np.bincount(np.arange(span) % n, minlength=n)
print(f"  reducing 0..{span - 1} by {n}: counts {counts}, "
      f"largest over smallest {counts.max() / counts.min():.4f}")
print(f"  the relative bias is about n / M = {n / span:.4f}")
```

The remainder pattern repeats with period $8$, and the bias in the small example is exactly the
predicted $n/M$.

### 1.5 What the inverse transform needs

It needs $F^{-1}$ in closed form, or at least computable cheaply and accurately.

Given that, the recipe is one line: if $u$ is uniform on $(0,1)$ then $F^{-1}(u)$ has distribution
$F$, for **any** distribution function $F$, continuous or not. Exercise 2.3 proves it including the
discontinuous case.

**When it is not available**, three routes.

*Find another identity.* The normal has no closed form inverse, but two uniforms map to two normals
through Box-Muller. The chi-square, the gamma at integer shape, and the beta at integer parameters
all have constructions of this kind.

*Reject.* Sample from something easy that dominates the target and throw away the excess, which is
Marsaglia's polar method and, more generally, the acceptance-rejection method. The cost is the
acceptance rate, which the lesson measures at exactly $\pi/4$ for the polar method.

*Invert numerically.* Tabulate $F$ and interpolate, or solve $F(x) = u$ by Newton's method from
lesson 12. That is what most libraries do for distributions with no better route, and its accuracy is
the accuracy of the root find rather than of the generator.

### 2.1 The Hull-Dobell conditions are sufficient

**Claim.** If $\gcd(c, m) = 1$, every prime factor of $m$ divides $a-1$, and $4 \mid m$ implies
$4 \mid a-1$, then the LCG has period $m$ from every seed.

**Setup.** Iterating the recurrence,

$$
x_k = a^{k}x_0 + c\,\frac{a^{k}-1}{a-1} \pmod m ,
$$

where the fraction is an integer because $a-1$ divides $a^{k}-1$. The period is the smallest $k > 0$
with $x_k = x_0$ for all $x_0$, that is the smallest $k$ with

$$
(a^{k}-1)\,x_0 + c\,\frac{a^{k}-1}{a-1} \equiv 0 \pmod m \quad\text{for every } x_0 .
$$

Taking $x_0 = 0$ and $x_0 = 1$ and subtracting shows the condition splits into $a^{k} \equiv 1$ and
$c(a^{k}-1)/(a-1) \equiv 0$, both mod $m$.

**The prime power case.** By the Chinese remainder theorem it suffices to prove the period is
$p^{e}$ for each prime power $p^{e} \Vert m$, since the overall period is the least common multiple.

Write $a = 1 + d$ with $p \mid d$, which is the second condition. Then

$$
\frac{a^{k}-1}{a-1} = \frac{(1+d)^{k}-1}{d} = k + \binom{k}{2}d + \binom{k}{3}d^{2} + \cdots
\equiv k \pmod p ,
$$

because every term after the first carries a factor of $d$. So the condition
$c(a^{k}-1)/(a-1) \equiv 0 \pmod{p}$ becomes $ck \equiv 0$, and since $\gcd(c, p) = 1$ by the first
condition, $p \mid k$. Repeating the argument modulo $p^{2}, \dots, p^{e}$, using $p \mid d$ each
time to keep the leading term, forces $p^{e} \mid k$.

**The factor of four.** At $p = 2$ the expansion above has $\binom{k}{2}d = k(k-1)d/2$, whose factor
of $1/2$ can cancel one power of two and break the induction at the second step. Requiring
$4 \mid d$ restores it, which is the third condition.

Since the period divides $m$ and is divisible by every $p^{e} \Vert m$, it is $m$. $\square$

```python
from nalib import rng as rg

out = rg.the_hull_dobell_conditions_are_exact()
print(f"modulus {out['modulus']}: {out['combinations']} combinations of a and c")
print(f"  the theorem and the measured period agree on {out['agreements']}")
print(f"  disagreements: {out['disagreements']}")
print(f"  full period reached in {out['full_period_count']} of them")

print(f"\n{'a':>5}{'c':>5}{'period':>9}{'predicted full':>17}{'agrees':>9}")
for a, c in ((1, 1), (5, 1), (5, 3), (13, 7), (13, 8), (21, 1), (64, 1), (33, 5)):
    measured = rg.period(a, c, 64)
    predicted = rg.hull_dobell(a, c, 64)
    print(f"{a:>5}{c:>5}{measured:>9}{str(predicted):>17}"
          f"{str((measured == 64) == predicted):>9}")

assert out["the_conditions_are_exact"]
```

The theorem is an equivalence and the sweep is exhaustive at this modulus, so a single disagreement
anywhere would disprove it. There are none in $4096$ cases.

### 2.2 The RANDU relation

RANDU has $a = 65539 = 2^{16} + 3$, $c = 0$, $m = 2^{31}$.

$$
a^{2} = (2^{16}+3)^{2} = 2^{32} + 6\cdot 2^{16} + 9 .
$$

Now $2^{32} = 2\cdot 2^{31} \equiv 0 \pmod{2^{31}}$, so

$$
a^{2} \equiv 6\cdot 2^{16} + 9 \pmod{2^{31}} .
$$

Rewrite the right side in terms of $a$: since $2^{16} = a - 3$,

$$
6\cdot 2^{16} + 9 = 6(a-3) + 9 = 6a - 18 + 9 = 6a - 9 .
$$

Therefore $a^{2} \equiv 6a - 9$, and multiplying by $x_k$,

$$
x_{k+2} \equiv a^{2}x_k \equiv 6a\,x_k - 9\,x_k \equiv 6\,x_{k+1} - 9\,x_k \pmod{2^{31}} ,
$$

which is

$$
\boxed{\;9\,x_k - 6\,x_{k+1} + x_{k+2} \equiv 0 \pmod{2^{31}}\;}
$$

The plane normal is $(9, -6, 1)$, of length $\sqrt{118} = 10.8628$. Since each $x$ is below $2^{31}$,
the combination lies between $-6\cdot 2^{31}$ and $10\cdot 2^{31}$, so it can take at most $16$
multiples of the modulus and the observed count is $15$.

```python
import math

import numpy as np
from nalib import rng as rg

a, c, m = rg.FAMOUS["randu"]
print(f"a = {a} = 2**16 + {a - 2 ** 16}")
print(f"a**2 mod m         = {a * a % m}")
print(f"6a - 9             = {6 * a - 9}")
print(f"they agree: {a * a % m == 6 * a - 9}")

found = rg.shortest_relation(a, m)
print(f"\nthe search finds {found['normal']}, of length {found['length']:.4f}")
print(f"sqrt(9**2 + 6**2 + 1) = {math.sqrt(81 + 36 + 1):.4f}")

out = rg.randu_lies_on_a_few_planes()
print(f"\nplanes counted: {out['planes']}, and the bound from the normal is "
      f"{9 + 6 + 1}")
print(f"the relation holds exactly, worst residual {out['worst_residual']}")

assert out["worst_residual"] == 0
assert found["normal"] == [9, -6, 1]
```

The identity $a^2 \bmod m = 6a - 9$ holds exactly in integer arithmetic, the search recovers the
normal without being told it, and the plane count of $15$ sits just under the bound of $16$ that the
normal's coefficients give.

### 2.3 The inverse transform

**Claim.** If $u$ is uniform on $(0,1)$ and $F$ is any distribution function, then
$X = F^{-1}(u)$ has distribution $F$, where

$$
F^{-1}(t) = \inf\{x : F(x) \ge t\}
$$

is the generalized inverse, defined whether or not $F$ is strictly increasing or continuous.

**Proof.** The key is the equivalence

$$
F^{-1}(t) \le x \quad\Longleftrightarrow\quad t \le F(x) .
$$

*Forward.* If $F^{-1}(t) \le x$ then, since $F$ is right continuous and non-decreasing, the infimum
defining $F^{-1}(t)$ is attained, so $F(F^{-1}(t)) \ge t$ and $F(x) \ge F(F^{-1}(t)) \ge t$.

*Backward.* If $t \le F(x)$ then $x$ belongs to the set $\{y : F(y) \ge t\}$, so the infimum of that
set is at most $x$, that is $F^{-1}(t) \le x$.

Given the equivalence,

$$
\Pr[X \le x] = \Pr[F^{-1}(u) \le x] = \Pr[u \le F(x)] = F(x) ,
$$

the last step because $u$ is uniform on $(0,1)$ and $F(x) \in [0,1]$. $\square$

**What the general form buys.** A discrete distribution has a step function for $F$, and the
generalized inverse picks out the right atom: the interval of $u$ of length $p_j$ maps to the value
$x_j$. So the same one line recipe handles a die, a Poisson variable and a normal, and the only thing
that changes is how $F^{-1}$ is computed.

```python
import math

import numpy as np
from nalib import rng as rg

rng = np.random.default_rng(42)
draws = 400000
uniforms = rng.random(draws)

print("the continuous case: the exponential")
for rate in (0.5, 1.0, 4.0):
    values = rg.inverse_transform_exponential(uniforms, rate=rate)
    test = rg.kolmogorov_smirnov(values, lambda t, r=rate: 1.0 - math.exp(-r * t))
    print(f"  rate {rate:>4}: mean {float(np.mean(values)):.5f} against "
          f"{1.0 / rate:.5f}, scaled KS gap {test['scaled']:.4f}, "
          f"passes {test['passes_at_one_percent']}")

print("\nthe discrete case: a loaded die, through the same recipe")
weights = np.array([0.1, 0.05, 0.25, 0.3, 0.2, 0.1])
edges = np.cumsum(weights)
faces = np.searchsorted(edges, uniforms) + 1
counts = np.bincount(faces, minlength=weights.size + 1)[1:]
print(f"{'face':>6}{'wanted':>10}{'measured':>11}{'difference':>13}")
for face in range(weights.size):
    share = counts[face] / draws
    print(f"{face + 1:>6}{weights[face]:>10.4f}{share:>11.4f}"
          f"{share - weights[face]:>+13.5f}")
print(f"  the sampling bar on each share is about "
      f"{math.sqrt(0.25 / draws):.5f}")
```

Every continuous case passes at one per cent, and every face of the loaded die lands within the
sampling bar of its intended weight, from the same construction.

### 2.4 Box-Muller

**Setup.** Let $Z_1, Z_2$ be independent standard normals. Their joint density is

$$
\frac{1}{2\pi}e^{-(z_1^{2}+z_2^{2})/2} ,
$$

which depends on $(z_1, z_2)$ only through $r^{2} = z_1^{2}+z_2^{2}$. So in polar coordinates
$z_1 = r\cos\theta$, $z_2 = r\sin\theta$, with Jacobian $r\,dr\,d\theta$, the density becomes

$$
\frac{1}{2\pi}e^{-r^{2}/2}\,r\,dr\,d\theta .
$$

**It factors.** The $\theta$ part is $d\theta/2\pi$ on $[0, 2\pi)$, so $\theta$ is **uniform** and
independent of $r$. The $r$ part is $r e^{-r^{2}/2}dr$, and substituting $s = r^{2}/2$ gives
$e^{-s}ds$: **$R^{2}/2$ is exponential with rate 1.**

**Inverting each.** By 2.3, an exponential with rate $1$ is $-\log(1-u_1)$, so

$$
r = \sqrt{-2\log(1-u_1)} ,
$$

and a uniform angle is $\theta = 2\pi u_2$. Converting back,

$$
\boxed{\;z_1 = \sqrt{-2\log(1-u_1)}\,\cos(2\pi u_2), \qquad
z_2 = \sqrt{-2\log(1-u_1)}\,\sin(2\pi u_2)\;}
$$

Both are standard normal and they are **independent**, because $r$ and $\theta$ were.

Using $\log(1-u)$ rather than $\log u$ matters in floating point: $u$ can be exactly $0$, where the
logarithm is $-\infty$, and $1-u$ cannot reach $0$ in a generator that returns values below $1$.

```python
import math

import numpy as np
from nalib import rng as rg

rng = np.random.default_rng(42)
pairs = rng.random(2000000)
normals = rg.box_muller(pairs)
first, second = normals[0::2], normals[1::2]

print(f"mean     {float(np.mean(normals)):+.6f}, wanted 0")
print(f"variance {float(np.var(normals)):.6f}, wanted 1")
print(f"skewness {float(np.mean(normals ** 3)):+.6f}, wanted 0")
print(f"kurtosis {float(np.mean(normals ** 4)):.6f}, wanted 3")
print(f"\ncorrelation between the two of each pair: "
      f"{float(np.corrcoef(first, second)[0, 1]):+.6f}")
print(f"the radius squared over 2 should be exponential with rate 1:")
radius = 0.5 * (first ** 2 + second ** 2)
test = rg.kolmogorov_smirnov(radius, lambda t: 1.0 - math.exp(-t))
print(f"  mean {float(np.mean(radius)):.6f}, scaled KS gap {test['scaled']:.4f}, "
      f"passes {test['passes_at_one_percent']}")

out = rg.the_transforms_are_exact()
assert out["normal_passes"]
```

All four moments match, the two members of a pair are uncorrelated to five decimals, and $R^2/2$ is
exponential to the accuracy of a two million sample test. The derivation's three claims, that the
angle is uniform, that $R^2/2$ is exponential, and that the two are independent, are each checked
separately.

### 2.5 The period of bit $j$

**Claim.** For an LCG with modulus $m = 2^{e}$, bit $j$ of the state, counted from the least
significant, has period at most $2^{j+1}$.

**Proof.** Reduce the recurrence modulo $2^{j+1}$. Because $2^{j+1}$ divides $m$,

$$
x_{k+1} \bmod 2^{j+1} = \big((a x_k + c) \bmod 2^{e}\big) \bmod 2^{j+1}
= (a\,(x_k \bmod 2^{j+1}) + c) \bmod 2^{j+1} ,
$$

using that reduction mod $2^{e}$ then mod $2^{j+1}$ is the same as reduction mod $2^{j+1}$ directly,
and that multiplication and addition commute with reduction. So **the bottom $j+1$ bits form an LCG
of their own**, with modulus $2^{j+1}$.

That smaller generator has at most $2^{j+1}$ states, so its sequence repeats with period at most
$2^{j+1}$. Bit $j$ is a function of those bits alone, so its period divides that. $\square$

**And the bound is attained** when the Hull-Dobell conditions hold for $2^{j+1}$, which they do
whenever they hold for $2^{e}$, since the conditions only involve the prime $2$.

```python
import numpy as np
from nalib import rng as rg

out = rg.the_low_bits_are_worse()
print(f"modulus {out['modulus']}, whose own period is {out['modulus']}")
print(f"{'bit':>5}{'bound 2**(j+1)':>17}{'measured':>11}{'attained':>11}")
for row in out["rows"]:
    print(f"{row['bit']:>5}{row['predicted_period']:>17}{row['measured_period']:>11}"
          f"{str(row['matches']):>11}")

print("\nand the bottom bits really are their own generator:")
a, c, m = 1103515245, 12345, 2 ** 20
full = rg.lcg(64, a, c, m, seed=1)
small = rg.lcg(64, a, c, 2 ** 4, seed=1)
print(f"  the full run reduced mod 16: {(full % 16)[:16]}")
print(f"  a generator with modulus 16: {small[:16]}")
print(f"  they agree: {bool(np.all(full % 16 == small))}")

assert bool(np.all(full % 16 == small))
assert out["every_bit_matches"]
```

The bound is attained at every bit, and the last check is the proof made concrete: reducing the big
generator modulo $16$ gives **exactly** the sequence a generator with modulus $16$ produces.

### 3.1 The spectral test by lattice reduction

The bounded search of the lesson finds a short relation when one exists inside its box and reports
nothing otherwise. The proper test finds the **shortest** vector of the dual lattice, whatever its
size, by lattice reduction.

```python
import numpy as np
from nalib import rng as rg


def lll(basis, delta=0.75):
    """Lenstra-Lenstra-Lovasz reduction, enough for the small bases here."""
    rows = [np.array(row, dtype=float) for row in basis]
    n = len(rows)

    def gram_schmidt(current):
        star, mu = [], np.zeros((n, n))
        for i in range(n):
            v = current[i].copy()
            for j in range(i):
                mu[i, j] = float(current[i] @ star[j]) / float(star[j] @ star[j])
                v = v - mu[i, j] * star[j]
            star.append(v)
        return star, mu

    star, mu = gram_schmidt(rows)
    k = 1
    while k < n:
        for j in range(k - 1, -1, -1):
            if abs(mu[k, j]) > 0.5:
                rows[k] = rows[k] - round(mu[k, j]) * rows[j]
                star, mu = gram_schmidt(rows)
        if (float(star[k] @ star[k])
                >= (delta - mu[k, k - 1] ** 2) * float(star[k - 1] @ star[k - 1])):
            k += 1
        else:
            rows[k], rows[k - 1] = rows[k - 1], rows[k]
            star, mu = gram_schmidt(rows)
            k = max(k - 1, 1)
    return rows


def spectral(multiplier, modulus, dimension):
    """The shortest vector of the lattice the consecutive tuples lie on."""
    basis = [[modulus] + [0] * (dimension - 1)]
    for i in range(1, dimension):
        row = [0] * dimension
        row[0] = -pow(multiplier, i, modulus)
        row[i] = 1
        basis.append(row)
    reduced = lll(basis)
    best = min(reduced, key=lambda v: float(v @ v))
    return float(np.linalg.norm(best)), best


print(f"{'generator':>20}{'d':>4}{'shortest vector':>18}{'m**(1/d)':>12}{'ratio':>9}")
for name in ("randu", "minstd"):
    a, c, m = rg.FAMOUS[name]
    for dimension in (2, 3, 4):
        length, vector = spectral(a, m, dimension)
        print(f"{name:>20}{dimension:>4}{length:>18.2f}"
              f"{m ** (1.0 / dimension):>12.2f}{length / m ** (1.0 / dimension):>9.4f}")

length, vector = spectral(*rg.FAMOUS["randu"][::2], 3)
found = rg.shortest_relation(*rg.FAMOUS["randu"][::2])
print(f"\nthe reduction's shortest vector in three dimensions has length {length:.4f}")
print(f"the lesson's bounded search found {found['normal']} of length "
      f"{found['length']:.4f}")
print(f"they agree: {abs(length - found['length']) < 1e-9}")

assert abs(length - found["length"]) < 1e-9
```

**RANDU is the best possible generator in two dimensions and the worst in three.** Its ratio is
$0.9999$ at $d = 2$, essentially the theoretical maximum, and $0.0084$ at $d = 3$, a collapse by a
factor of $119$.

That single pair of numbers explains the whole history. Every test anyone ran on RANDU in the 1960s
was a one or two dimensional test, and in two dimensions it is not merely acceptable, it is
excellent.

The reduction's answer agrees with the lesson's bounded search to nine digits, which is the check
that the cheap method was finding the right vector and not merely a short one.

`minstd`'s worst dimension is $d = 2$, at $0.363$: not catastrophic, and its known weakness.

### 3.2 A lagged Fibonacci generator, and its own failure

$$
x_k = (x_{k-j} + x_{k-\ell}) \bmod m , \qquad j < \ell ,
$$

with $\ell$ values of state. Its period can be enormous, up to $(2^{\ell}-1)2^{e-1}$ for
$m = 2^{e}$, and its failure is not an LCG's.

```python
import numpy as np
from nalib import rng as rg


def lagged_fibonacci(count, short=7, long_lag=10, modulus=2 ** 32, seed=1):
    state = list(rg.lcg(long_lag, 1103515245, 12345, modulus, seed=seed))
    out = np.empty(count, dtype=np.int64)
    for i in range(count):
        value = (state[-short] + state[-long_lag]) % modulus
        state.append(value)
        state.pop(0)
        out[i] = value
    return out


print(f"{'lags':>12}{'chi-square ratio':>19}{'lag 1 correlation':>20}"
      f"{'triples on the relation':>26}")
for short, long_lag in ((7, 10), (5, 17), (24, 55)):
    raw = lagged_fibonacci(200000, short, long_lag)
    values = raw / 2 ** 32
    flat = rg.chi_square_uniformity(values, bins=100)
    combination = (raw[:-long_lag] + raw[long_lag - short:-short] - raw[long_lag:]) % 2 ** 32
    share = float(np.mean(combination == 0))
    print(f"{f'({short}, {long_lag})':>12}{flat['ratio']:>19.4f}"
          f"{rg.serial_correlation(values, 1):>+20.5f}{share:>26.4f}")

modern = (np.random.default_rng(42).random(200000) * 2 ** 32).astype(np.int64)
combination = (modern[:-10] + modern[3:-7] - modern[10:]) % 2 ** 32
print(f"\n{'pcg64':>12}{'':>19}{'':>20}{float(np.mean(combination == 0)):>26.6f}")
```

**Every lagged Fibonacci generator passes uniformity and the correlation test**, with ratios of
$0.93$ to $1.23$ and lag one correlations below $0.0021$. And **every single triple at lags
$(0, \ell-j, \ell)$ satisfies the defining relation exactly**, at a rate of $1.0000$, where a modern
generator gives $0.000000$.

That is a different failure from an LCG's, and the difference matters for testing. An LCG's
**consecutive** triples lie on few planes, so a three dimensional test on consecutive values finds
it. A lagged Fibonacci's consecutive triples are fine; it is the triples at the **lags** that
degenerate, and a test that does not know the lags will not look there.

The practical consequence is that a test battery has to include tests at many lags, which is what
Diehard's overlapping permutations and TestU01's collision tests are for. And a generator whose
construction is public has a known worst lag, which is one argument for preferring a design with no
such structure at all.

### 3.3 The ziggurat method

The ziggurat covers the normal density by a stack of equal area rectangles plus a tail piece. Almost
every draw needs one uniform, one table lookup and one comparison, with no transcendental function at
all; the expensive branches are taken a few per cent of the time.

```python
import math
import time

import numpy as np
from nalib import rng as rg


def build_ziggurat(layers=128):
    """Solve for the layer boundaries and the common area, by bisection on the closure.

    The stack is built from the tail edge inward. The right edge is correct when the topmost layer
    closes exactly, that is when ``f(x) + area/x = 1``, and the bisection drives that residual to
    rounding.
    """
    f = lambda x: math.exp(-0.5 * x * x)
    tail = lambda x: math.sqrt(math.pi / 2.0) * math.erfc(x / math.sqrt(2.0))

    def closure(right):
        area = right * f(right) + tail(right)
        x, edges = right, [right]
        for _ in range(layers - 1):
            top = f(x) + area / x
            if top >= 1.0:
                return 1.0, None, None, None
            x = math.sqrt(-2.0 * math.log(top))
            edges.append(x)
        return f(x) + area / x - 1.0, x, area, edges

    low, high = 1.0, 6.0
    for _ in range(300):
        middle = 0.5 * (low + high)
        miss, _, _, _ = closure(middle)
        if miss > 0.0:
            low = middle
        else:
            high = middle
    return (high,) + closure(high)[1:] + (closure(high)[0],)


print(f"{'layers':>8}{'tail edge':>13}{'innermost x':>14}{'layer area':>14}"
      f"{'closure miss':>15}{'covered area':>15}{'rejected':>11}")
under = math.sqrt(2.0 * math.pi) / 2.0
for layers in (32, 128, 256):
    right, innermost, area, edges, miss = build_ziggurat(layers)
    covered = layers * area
    print(f"{layers:>8}{right:>13.6f}{innermost:>14.6f}{area:>14.8f}"
          f"{miss:>+15.2e}{covered:>15.8f}{1.0 - under / covered:>11.4%}")
print(f"  the area under the half density is {under:.8f}")

draws = 2000000
print(f"\ncost per normal, {draws} of them:")
rows = []
for label, make in (
        ("Box-Muller", lambda: rg.box_muller(np.random.default_rng(1).random(draws))),
        ("polar rejection", lambda: rg.polar_normal(draws)["values"]),
        ("numpy, a ziggurat", lambda: np.random.default_rng(1).standard_normal(draws))):
    clock = time.perf_counter()
    values = make()
    elapsed = time.perf_counter() - clock
    test = rg.kolmogorov_smirnov(
        values[:400000], lambda t: 0.5 * (1.0 + math.erf(t / math.sqrt(2.0))))
    rows.append((label, elapsed, test["scaled"]))
    print(f"  {label:>20}: {elapsed:.4f} s, {1e9 * elapsed / draws:>6.1f} ns each, "
          f"scaled KS gap {test['scaled']:.4f}")
slowest = max(r[1] for r in rows)
print(f"\n  relative to the fastest: "
      + ", ".join(f"{r[0]} {r[1] / min(x[1] for x in rows):.2f}x" for r in rows))
```

The closure residual is at rounding, which is the check that the table solve converged.

**The covered area exceeds the area under the density, and that excess is the rejection rate.** The
rectangles cover the curve, so their total must be larger, and the gap is the fraction of draws that
fall between a rectangle's top and the curve and have to be handled by the slow branch. It is
$0.81$, $0.44$ and $0.28$ per cent at $32$, $128$ and $256$ layers, so **more layers buy a smaller
slow branch at the cost of a bigger table**, and $128$ is the usual compromise.

On this machine the ziggurat is about three times faster than either explicit transform, and all
three produce normals that pass at the same level.

The measurement also settles a common claim. **The polar method is not faster than Box-Muller
here**, losing by ten to fifteen per cent across runs, because the trigonometric calls are vectorized
and the rejection loop is not: a branch that discards a fifth of a vector costs more than two
vectorized cosines. On scalar hardware with slow trigonometry the ordering reverses, which is why
both survive in the literature. Absolute timings move by a few per cent between runs, so the ratios
are what to read here and not the nanoseconds.

### 3.4 The gap test and birthday spacings

**The gap test** watches how long the sequence goes between visits to an interval. If the values are
independent and the interval has width $p$, the gap lengths are geometric with parameter $p$, and a
chi-square against that distribution catches a generator whose values are locally clustered.

**Birthday spacings** draws $k$ values from $n$ cells, sorts them, and counts repeated spacings. The
number of repeats is approximately Poisson with mean $k^{3}/(4n)$, and it catches lattice structure
because a lattice makes certain spacings far more common than chance.

```python
import math

import numpy as np
from nalib import rng as rg


def gap_test(values, low=0.0, high=0.1, longest=20):
    inside = (np.asarray(values) >= low) & (np.asarray(values) < high)
    hits = np.flatnonzero(inside)
    gaps = np.minimum(np.diff(hits) - 1, longest)
    counts = np.bincount(gaps, minlength=longest + 1)
    share = high - low
    expected = counts.sum() * np.array(
        [share * (1.0 - share) ** k for k in range(longest)] + [(1.0 - share) ** longest])
    keep = expected > 5.0
    return (float(np.sum((counts[keep] - expected[keep]) ** 2 / expected[keep]))
            / max(int(keep.sum()) - 1, 1))


def birthday_spacings(values, cells=2 ** 20, drawn=512, rounds=200):
    v = np.asarray(values)
    repeats = []
    for r in range(rounds):
        block = v[r * drawn:(r + 1) * drawn]
        if block.size < drawn:
            break
        sorted_cells = np.sort((block * cells).astype(np.int64))
        spacings = np.diff(sorted_cells)
        repeats.append(int(spacings.size - np.unique(spacings).size))
    return float(np.mean(repeats)), drawn ** 3 / (4.0 * cells)


print(f"{'generator':>20}{'gap ratio':>12}{'birthday mean':>16}{'predicted':>12}"
      f"{'ratio':>9}")
for name in sorted(rg.FAMOUS):
    values = rg.lcg_uniform(200000, name)
    mean, predicted = birthday_spacings(values)
    print(f"{name:>20}{gap_test(values):>12.4f}{mean:>16.3f}{predicted:>12.3f}"
          f"{mean / predicted:>9.3f}")
values = np.random.default_rng(42).random(200000)
mean, predicted = birthday_spacings(values)
print(f"{'pcg64':>20}{gap_test(values):>12.4f}{mean:>16.3f}{predicted:>12.3f}"
      f"{mean / predicted:>9.3f}")
```

**Neither test finds RANDU.** The gap ratios run from $0.54$ to $1.17$ with RANDU at $0.997$, which
is the best value in the table, and every birthday ratio including RANDU's sits at $0.93$ to $0.95$.

That is a negative result and it is the point. The exercise asks for a generator each test catches
that the others do not, and on this set of generators the answer is that **these two tests catch
nothing at all**. They are in the standard batteries because they catch other kinds of failure, not
because they are stronger than the spectral test.

One honest caveat about the birthday column. Every ratio is about $0.94$, including the modern
generator's, so the six per cent gap is in the Poisson approximation rather than in any generator: at
$k = 512$ and $n = 2^{20}$ the mean is $32$, which is large enough for the approximation to be
visibly imperfect. A test that flagged $0.94$ as a failure would flag everything.

### 3.5 A counter based generator

An LCG's state advances by iteration, so jumping ahead means iterating. A **counter based** generator
computes its output as a function of a counter and a key,

$$
x_n = f(n, \text{key}) ,
$$

with $f$ a small block cipher. Nothing is iterated, so any output can be produced directly, and two
workers with different keys produce independent streams by construction.

```python
import numpy as np
from nalib import rng as rg

MASK = (1 << 64) - 1
MULTIPLIERS = (0xD2B74407B1CE6E93, 0xCA5A826395121157)
BUMPS = (0x9E3779B97F4A7C15, 0xBB67AE8584CAA73B)


def philox_round(counter, key):
    """One round of a Philox style mix: two multiply-high-low steps and a swap."""
    lo0, hi0 = counter[0] * MULTIPLIERS[0] & MASK, (counter[0] * MULTIPLIERS[0]) >> 64
    lo1, hi1 = counter[1] * MULTIPLIERS[1] & MASK, (counter[1] * MULTIPLIERS[1]) >> 64
    return (hi1 ^ counter[2] ^ key[0], lo1, hi0 ^ counter[3] ^ key[1], lo0)


def counter_based(count, key=(0, 0), start=0, rounds=10):
    out = np.empty(count)
    for i in range(count):
        n = start + i
        block = (n & MASK, (n >> 64) & MASK, 0x243F6A8885A308D3, 0x13198A2E03707344)
        current = tuple(key)
        for _ in range(rounds):
            block = philox_round(block, current)
            current = ((current[0] + BUMPS[0]) & MASK, (current[1] + BUMPS[1]) & MASK)
        out[i] = (block[0] ^ block[2]) / 2.0 ** 64
    return out


first = counter_based(20000, key=(1, 0))
second = counter_based(20000, key=(2, 0))
third = counter_based(20000, key=(1, 0), start=1000000)

print(f"stream with key 1: chi-square ratio "
      f"{rg.chi_square_uniformity(first, bins=50)['ratio']:.4f}, "
      f"lag 1 {rg.serial_correlation(first, 1):+.5f}")
print(f"stream with key 2: chi-square ratio "
      f"{rg.chi_square_uniformity(second, bins=50)['ratio']:.4f}, "
      f"lag 1 {rg.serial_correlation(second, 1):+.5f}")
print(f"\ncorrelation between the two keys:        "
      f"{float(np.corrcoef(first, second)[0, 1]):+.6f}")
print(f"correlation between two counter blocks:  "
      f"{float(np.corrcoef(first, third)[0, 1]):+.6f}")
print(f"the sampling bar is about {1.0 / np.sqrt(first.size):.6f}")

print(f"\nand jumping ahead is free: element 1000000 computed directly")
print(f"  {counter_based(1, key=(1, 0), start=1000000)[0]:.10f}")
print(f"  matches the block above: "
      f"{abs(counter_based(1, key=(1, 0), start=1000000)[0] - third[0]) == 0.0}")
```

Both streams pass uniformity, and the correlation between different keys and between different
counter blocks is at the sampling bar. **Jumping to element one million costs the same as computing
element one**, which no iterated generator can do.

That property is why counter based designs dominate on GPUs, where a million threads each need their
own stream and none of them can afford to iterate a shared state.

### 4.1 The spectral test across dimensions

```python
import numpy as np
from nalib import rng as rg

print(f"{'generator':>20}" + "".join(f"{f'd={d}':>10}" for d in range(2, 9)))
worst = {}
for name in sorted(rg.FAMOUS):
    a, c, m = rg.FAMOUS[name]
    row = []
    for dimension in range(2, 9):
        length, _ = spectral(a, m, dimension)
        row.append(length / m ** (1.0 / dimension))
    worst[name] = (min(row), range(2, 9)[int(np.argmin(row))])
    print(f"{name:>20}" + "".join(f"{v:>10.4f}" for v in row))

print(f"\n{'generator':>20}{'worst ratio':>14}{'at dimension':>15}")
for name in sorted(worst):
    print(f"{name:>20}{worst[name][0]:>14.4f}{worst[name][1]:>15}")
```

The column is the shortest dual vector divided by $m^{1/d}$, which is $1$ for the best lattice
possible at that density and near $0$ for a badly structured one.

**RANDU is the only generator with a catastrophic dimension, and it is exactly three.** Its ratios
are $0.9999, 0.0084, 0.0500, 0.1465, 0.2999, 0.5001, 0.7341$: perfect at two, a collapse by a factor
of $119$ at three, and a slow recovery afterwards.

**Every other generator stays between $0.36$ and $1.07$ at every dimension tested.** `minstd`'s
worst is $0.363$ at $d = 2$ and `numerical recipes` never falls below $0.75$.

So the answer to where each first looks bad: RANDU at three, `minstd` at two, and the other three
never, on this test up to eight dimensions.

The value above one for `numerical recipes` at $d = 2$ is not an error. The normalization
$m^{1/d}$ is the density of the lattice, and the best achievable ratio depends on the optimal packing
in that dimension, which exceeds one in two dimensions by the hexagonal packing constant.

### 4.2 How many samples it takes to see the failure

```python
import math

import numpy as np
from nalib import rng as rg


def cube_ratio(values, side):
    triples = np.column_stack([values[:-2], values[1:-1], values[2:]])
    cells = np.minimum((triples * side).astype(int), side - 1)
    index = (cells[:, 0] * side + cells[:, 1]) * side + cells[:, 2]
    counts = np.bincount(index, minlength=side ** 3)
    expected = index.size / side ** 3
    return float(np.sum((counts - expected) ** 2 / expected)) / (side ** 3 - 1)


bad = rg.lcg_uniform(2000000, "randu")
good = np.random.default_rng(42).random(2000000)
budgets = (2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000,
           1000000, 2000000)
print(f"{'cells per axis':>16}{'cells':>9}{'first N that fails':>21}"
      f"{'ratio there':>14}{'PCG64 there':>14}")
for side in (4, 6, 8, 10, 14, 20):
    bar = 1.0 + 6.0 / math.sqrt(side ** 3 - 1)
    found = None
    for n in budgets:
        if n < 10 * side ** 3:
            continue
        ratio = cube_ratio(bad[:n], side)
        if ratio > bar:
            found = (n, ratio, cube_ratio(good[:n], side))
            break
    if found:
        print(f"{side:>16}{side ** 3:>9}{found[0]:>21}{found[1]:>14.4f}"
              f"{found[2]:>14.4f}")
    else:
        print(f"{side:>16}{side ** 3:>9}{'never, up to 2e6':>21}")
```

Two findings, and the second is the interesting one.

**More cells make the failure easier to see.** At $20$ cells per axis the ratio is $10.06$ and at
$4$ it is $2.02$, because finer cells resolve the fifteen planes and most cells end up empty. The
sample count needed does not change much, $50000$ in most rows, because the statistic grows faster
than the noise does.

**At six cells per axis the failure is invisible, even at two million samples.** That is not a
sample size problem. With $216$ cells and $15$ planes the plane spacing happens to distribute the
points across the cells almost evenly, so the counts come out right and the test passes.

The lesson is a real one about testing. **A grid that happens to align with the structure hides it**,
so a single cell count is not a test, and a battery has to sweep the resolution as well as the sample
size. The same phenomenon is why TestU01 runs its collision tests at several densities rather than
one.

### 4.3 The cost of a normal

```python
import math
import time

import numpy as np
from nalib import rng as rg

draws = 2000000
rows = []
for label, make in (
        ("Box-Muller", lambda: rg.box_muller(np.random.default_rng(1).random(draws))),
        ("polar rejection", lambda: rg.polar_normal(draws)["values"]),
        ("numpy standard_normal", lambda: np.random.default_rng(1).standard_normal(draws)),
        ("uniforms alone", lambda: np.random.default_rng(1).random(draws))):
    clock = time.perf_counter()
    values = make()
    elapsed = time.perf_counter() - clock
    rows.append((label, elapsed, 1e9 * elapsed / draws))

print(f"{'method':>26}{'seconds':>11}{'ns per value':>15}{'over the uniform':>19}")
base = rows[-1][2]
for label, elapsed, each in rows:
    print(f"{label:>26}{elapsed:>11.4f}{each:>15.1f}{each / base:>19.2f}")

polar = rg.the_polar_method_wastes_a_known_fraction()
print(f"\nthe polar method draws {1.0 / polar['acceptance']:.4f} uniforms per accepted "
      f"pair, against a predicted {4.0 / math.pi:.4f}")
```

The cost breaks into three parts, and naming them explains the ranking.

**The uniforms themselves** are the floor: every method needs at least one per value, and that is
what the last row measures.

**Box-Muller** costs a logarithm, a square root and two trigonometric calls per pair, which
measures at about seven times the cost of the uniform alone.

**The polar method** replaces the trigonometry with a rejection loop that draws $1.2733$ uniforms
per accepted pair, against the predicted $4/\pi = 1.2732$, agreeing to five digits. It measures
**slower** than Box-Muller, at about eight and a half times the uniform.

**The ziggurat**, which is what `numpy` uses, costs one uniform, a table lookup and a comparison in
the common case. It measures at about two and a half times the uniform, roughly three times faster
than either explicit transform.

The polar method losing is the result worth keeping, because the usual claim is the reverse. It loses
because the trigonometric calls are vectorized and the rejection loop is not: a branch that discards
a fifth of a vector costs more than two vectorized cosines. On scalar hardware with slow
trigonometry the ordering reverses, and **which method is faster is a property of the machine rather
than of the mathematics.**

### 5.1 Why the period is not the point

The Mersenne Twister has period $2^{19937}-1$, a number with six thousand digits, and it still fails.

**Its weakness is initialization.** The state is $624$ words, and a seed of one word has to be
expanded to fill it. If the expansion leaves the state with many zero bits, the generator takes a
very long time to recover, because its transition is a **linear** map over $\mathbb{F}_2$ and a
sparse state stays sparse for thousands of outputs. The known figure is that a state with a single
set bit needs on the order of $10^{5}$ outputs before the output looks random.

**And there is a deeper one.** Because the transition is linear over $\mathbb{F}_2$, the bits of the
output satisfy linear recurrences, and TestU01's linear complexity and matrix rank tests detect them
regardless of the period. The Mersenne Twister fails those two tests at any sample size. It is not a
subtle failure that more samples would reveal; it is a structural property.

**What fixed it.** Three separate things, in the designs that followed.

*Better seeding.* MT19937's own initializer was replaced with one that mixes the seed through a
multiplicative recurrence, and the practice of seeding from an entropy pool rather than a counter
removed the sparse state case in practice.

*Output mixing.* PCG64 applies a nonlinear permutation to the output of a linear generator, which
destroys the linear relations while keeping the period and the cheap state update. That is why the
same tests it would otherwise fail are passed.

*Different algebra.* Counter based generators of exercise 3.5 use a block cipher, which has no linear
structure to find in the first place.

The general lesson is that **period is a necessary condition and a very weak one.** A generator with
period $2^{19937}$ and a linear output is worse for Monte Carlo than one with period $2^{128}$ and a
mixed output, and the reason is that a statistical test looks at structure and not at length.

### 5.2 Independent streams

Two workers need randomness that is independent, and there are two reliable ways and one common wrong
one.

**Splitting the period.** Advance a single generator by a huge fixed offset for each worker, so
worker $i$ starts at position $i \cdot 2^{64}$ of the same sequence. This needs a **jump ahead**
operation that skips $n$ steps in $O(\log n)$ time rather than $O(n)$, which every modern design
provides: for an LCG it is a closed form, and for PCG64 and the Mersenne Twister it is a polynomial
computation over the state.

*What it guarantees*: the streams are disjoint segments of one sequence, so they are as independent
as that sequence's own distant values are. *What it costs*: the jump operation, and the need to know
in advance how much of the period each worker may consume.

**Distinct keys.** A counter based generator takes a key, and different keys give different functions
entirely. Exercise 3.5 measures correlations at the sampling bar between two keys.

*What it guarantees*: independence in the sense the underlying cipher provides, which is stronger
than segment disjointness. *What it costs*: nothing; there is no coordination at all, which is why
this is the GPU answer.

**The wrong way is seeding each worker with a different small number.** Seeds $1, 2, 3, \dots$ are
nearby states, and for many generators nearby states produce correlated streams: an LCG seeded at $s$
and at $s+1$ produces sequences that differ by a fixed multiplicative pattern, and the Mersenne
Twister's sparse initialization means small seeds share most of their state. The streams are
**different** and they are not **independent**, which is exactly the failure that is hardest to
notice, because every stream passes every one dimensional test.

The modern repair, and what `numpy.random.SeedSequence` does, is to hash the worker index through a
strong mixing function before it becomes a seed, so that consecutive indices give states that are far
apart in every sense a test can measure.

### 5.3 Statistically good against cryptographically secure

**Statistically good** means: no test in a battery distinguishes the output from independent uniforms
at the sample sizes tested. That is a statement about a fixed set of tests and a fixed budget.

**Cryptographically secure** means: no polynomial time algorithm distinguishes the output from
independent uniforms with non-negligible advantage, **even given** arbitrarily many previous outputs.
That is a statement about all efficient algorithms, and it implies unpredictability of the next output
from every previous one.

**The gap between them is enormous.** PCG64 is statistically excellent and trivially insecure: given
a few outputs, its state can be recovered by solving a small system, and every future output follows.
The Mersenne Twister is worse still, since $624$ outputs determine the state exactly by linear
algebra. Neither is a defect for their purpose.

**Why Monte Carlo needs only the first.** A simulation is not an adversary. It computes an average of
a function that was chosen before the generator ran, and the only way the generator can spoil it is
if that particular function correlates with the generator's structure. That is what a test battery
checks for, and it is a far weaker requirement than defeating every efficient attack.

**The cost of the stronger property** is speed and, sometimes, reproducibility. A cryptographic
generator runs several times slower, and the strong ones deliberately draw from operating system
entropy so that two runs do not agree, which destroys the reproducibility every measurement in this
repository relies on.

There is one case where the distinction bites in numerical work: **when the function being averaged
depends on the samples already drawn**, as in an adaptive method or a simulation whose next step is
chosen from previous results. That is closer to an adversary, and it is where a statistically good
generator's structure can in principle be exploited by the algorithm itself. It has been observed in
practice for lattice generators in Markov chain Monte Carlo, and it is a further argument for a
generator with no exploitable structure rather than one that merely passes the tests anyone has
written down.

---

## Lesson 92, Monte Carlo Methods

### 1.1 Why the error does not depend on the dimension

The estimator is an average of $N$ independent copies of the single random variable $f(U)$, where
$U$ is uniform on the cube. Its variance is

$$
\operatorname{Var}\left(\frac1N\sum_k f(U_k)\right) = \frac{\operatorname{Var} f(U)}{N}
= \frac{\sigma^{2}}{N} ,
$$

so the standard deviation is $\sigma/\sqrt N$.

**The dimension appears nowhere in that calculation.** It enters only through $\sigma$, which is a
number attached to the function:

$$
\sigma^{2} = \int_{[0,1]^d} f(x)^{2}dx - \left(\int_{[0,1]^d} f(x)\,dx\right)^{2} .
$$

A twenty dimensional integrand can have a smaller $\sigma$ than a one dimensional one, and often
does.

The contrast with a grid is where the reason becomes clear. **A grid has to cover the cube**, and the
number of points needed to cover it to a given resolution grows like $n^{d}$. **A sample does not
cover anything.** It estimates an average, and the accuracy of an average depends on how many
independent observations there are and how variable they are, not on the geometry of the space they
live in.

### 1.2 The standard error, and what it estimates

$$
\text{standard error} = \frac{s}{\sqrt N} , \qquad
s^{2} = \frac{1}{N-1}\sum_k \big(f(U_k) - \bar f\big)^{2} .
$$

**It estimates the standard deviation of the estimator**, that is, how far the computed average would
typically be from the true integral if the whole calculation were repeated with fresh samples.

Three things it is not.

It is not a bound. About $32$ per cent of runs land further away than this, and about $0.3$ per cent
land more than three times further.

It is not exact. $s$ is itself estimated from the same sample, so the bar has its own error of
relative size $1/\sqrt{2N}$.

It is not meaningful when $\sigma$ is infinite. Exercise 5.2 constructs that case, where the formula
returns a finite number that means nothing.

What makes it a **confidence interval** rather than merely a scale is the central limit theorem: the
distribution of the error is approximately normal, so a one bar interval covers about $68.3$ per cent
of runs. How many samples that approximation needs is exercise 4.1, and the answer depends on the
skewness of the integrand.

### 1.3 When antithetic variates help

The estimator averages $f(u)$ and $f(1-u)$ over $N/2$ pairs. Comparing at equal function evaluations,
the variance ratio against plain sampling is $1/(1+\rho)$ with $\rho$ the correlation between the two
members of a pair.

**They help when $\rho < 0$**, that is when $f$ is decreasing in the reflection: a monotone integrand
gives $\rho$ close to $-1$ and the saving can be unbounded.

**They do nothing when $\rho = 0$**, which happens when $f(u)$ and $f(1-u)$ are uncorrelated.

**They cost when $\rho > 0$.** If $f$ is symmetric about the centre, $f(1-u) = f(u)$ identically, so
$\rho = +1$, the ratio is $1/2$, and the second evaluation of every pair is wasted: **the technique
halves the effective sample.**

The lesson measures all three: $\rho = -1$ with an exact answer, $\rho = -0.760$ with a saving of
$4.165$, and $\rho = +1$ with a measured ratio of $0.504$.

The practical test is cheap. Draw a few hundred pairs, compute the correlation, and decide before
committing to the technique.

### 1.4 What a control variate needs

It needs a function $g$ with two properties: **its integral must be known exactly**, and it must
**correlate with $f$**.

The variance ratio is $1/(1-\rho^{2})$, so the second requirement is quantitative and the first is
absolute. A control whose integral is only approximately known biases the estimator by that
approximation error, which does not shrink with $N$ and therefore eventually dominates.

**What it does not need**, and this is the part that is usually stated too weakly:

It does not need to be cheap, only cheaper than $f$. It does not need to be similar to $f$ in any
visual or structural sense; a function that happens to correlate at $0.99$ works exactly as well as
one that "looks like" $f$. It does not need a positive correlation, since $\rho^2$ appears. And it
does not need the coefficient $\beta$ to be known in advance, because $\beta$ is estimated from the
same sample at negligible cost.

The lesson measures three controls of increasing quality and finds the predicted ratio to zero
relative error each time. **Searching for a control variate is searching for correlation and nothing
else.**

### 1.5 Low discrepancy against random

**Discrepancy** measures how far a point set is from perfectly even. For the star discrepancy, take
every box anchored at the origin, compare the fraction of points inside it against its volume, and
take the largest gap:

$$
D_N^{*} = \sup_{\text{boxes } B} \left| \frac{\#\{x_k \in B\}}{N} - \operatorname{vol}(B)\right| .
$$

A **low discrepancy sequence** is one whose $D_N^{*}$ falls like $(\log N)^{d}/N$, rather than the
$N^{-1/2}$ of a random sample.

**It is not random, and does not try to be.** A random sample clumps and leaves gaps, and both are of
size $N^{-1/2}$ by construction: that is what independence produces. A low discrepancy sequence is
deliberately constructed so that each new point falls in the largest remaining gap, so consecutive
points are strongly **negatively** correlated. Every test in lesson 91 that a good generator passes,
a Halton sequence fails, and that is intended.

The consequence is that it has no error bar. A random sample supports the central limit theorem and
therefore a confidence interval; a deterministic point set supports only Koksma's inequality, which
needs the variation of $f$ and is usually far too pessimistic to be useful. Exercise 5.3 is the
standard repair.

### 2.1 The standard error and the interval

**The variance.** For independent $X_k = f(U_k)$ with common variance $\sigma^{2}$,

$$
\operatorname{Var}\left(\frac1N\sum X_k\right) = \frac{1}{N^{2}}\sum\operatorname{Var}(X_k)
= \frac{N\sigma^{2}}{N^{2}} = \frac{\sigma^{2}}{N} .
$$

Independence is used once, to let the variances add. Nothing else about the $X_k$ matters.

**Why $s^2$ has $N-1$.** The sample mean is itself estimated from the data, so the sum of squared
deviations about it is systematically small. Dividing by $N-1$ instead of $N$ corrects the bias
exactly: $\mathbb{E}[s^{2}] = \sigma^{2}$.

**Which theorem makes it an interval.** The **central limit theorem**: for independent identically
distributed $X_k$ with finite variance,

$$
\frac{\bar X - \mu}{\sigma/\sqrt N} \longrightarrow N(0, 1) \quad\text{in distribution.}
$$

Replacing $\sigma$ by $s$ is legitimate by Slutsky's theorem, since $s \to \sigma$ in probability. So
$\bar X \pm z\,s/\sqrt N$ covers $\mu$ with the probability the normal assigns to $[-z, z]$, in the
limit.

**The hypothesis that matters is the finite variance.** Without it the theorem does not apply, the
limit is a stable law with heavier tails, and the interval means nothing. That is exercise 5.2.

```python
import math

import numpy as np
from nalib import montecarlo as mc

out = mc.the_error_bar_is_honest()
print(f"{out['repeats']} runs of {out['samples']} samples, {out['dimension']} dimensions")
print(f"  spread of the estimates:      {out['actual_spread']:.6f}")
print(f"  mean reported standard error: {out['claimed_spread']:.6f}")
print(f"  ratio:                        {out['ratio']:.4f}")
print(f"  coverage: {out['fraction_within_one_bar']:.3f} against "
      f"{out['predicted_fraction']:.3f}")

print("\nthe 1/(N-1) matters, and here is the size of the effect:")
rng = np.random.default_rng(42)
for n in (3, 10, 100):
    draws = rng.standard_normal((200000, n))
    biased = float(np.mean(np.var(draws, axis=1, ddof=0)))
    unbiased = float(np.mean(np.var(draws, axis=1, ddof=1)))
    print(f"  N = {n:>4}: dividing by N gives {biased:.5f}, by N-1 gives "
          f"{unbiased:.5f}, and (N-1)/N = {(n - 1) / n:.5f}")

assert out["the_bar_is_honest"]
```

The bar matches the spread to two per cent, the coverage matches the normal's, and the $N-1$
correction is exactly the factor $(N-1)/N$ at every sample size.

### 2.2 The antithetic variance ratio

Let $A = f(U)$ and $B = f(1-U)$, with common variance $\sigma^{2}$ and correlation $\rho$. The paired
value is $P = \tfrac12(A+B)$, whose variance is

$$
\operatorname{Var}(P) = \tfrac14\big(\operatorname{Var}A + \operatorname{Var}B
+ 2\operatorname{Cov}(A,B)\big)
= \tfrac14\big(2\sigma^{2} + 2\rho\sigma^{2}\big)
= \frac{\sigma^{2}(1+\rho)}{2} .
$$

**Now count evaluations.** A budget of $N$ evaluations buys $N/2$ pairs, so the antithetic estimator
is an average of $N/2$ copies of $P$:

$$
\operatorname{Var}(\text{antithetic}) = \frac{\operatorname{Var}(P)}{N/2}
= \frac{2}{N}\cdot\frac{\sigma^{2}(1+\rho)}{2} = \frac{\sigma^{2}(1+\rho)}{N} .
$$

The plain estimator with the same budget has variance $\sigma^{2}/N$. Therefore

$$
\boxed{\;\frac{\operatorname{Var}(\text{plain})}{\operatorname{Var}(\text{antithetic})}
= \frac{1}{1+\rho}\;}
$$

**The two factors of two are the whole subtlety.** Comparing per **pair** rather than per evaluation
gives $2/(1+\rho)$, which is twice as flattering and is the form that appears when the accounting is
done carelessly. At $\rho = +1$ the correct ratio is $1/2$, a loss; the careless one is $1$, no
change.

```python
import numpy as np
from nalib import montecarlo as mc

out = mc.antithetic_helps_only_a_monotone_integrand()
print(f"{'integrand':>36}{'rho':>12}{'1/(1+rho)':>13}{'measured':>12}{'ratio':>9}")
for row in out["rows"]:
    predicted = row["predicted_variance_ratio"]
    shown = "infinite" if predicted == float("inf") else f"{predicted:.4f}"
    quality = ("-" if predicted == float("inf")
               else f"{row['measured_variance_ratio'] / predicted:.4f}")
    print(f"{row['integrand']:>36}{row['correlation']:>+12.6f}{shown:>13}"
          f"{row['measured_variance_ratio']:>12.4f}{quality:>9}")
print(f"\nworst relative miss: {out['worst_relative_miss']:.2e}")

print("\nand the careless accounting, per pair instead of per evaluation:")
for row in out["rows"]:
    if row["predicted_variance_ratio"] != float("inf"):
        print(f"  {row['integrand']:>36}: correct "
              f"{row['predicted_variance_ratio']:.4f}, per pair "
              f"{2.0 * row['predicted_variance_ratio']:.4f}")

assert out["the_prediction_holds"]
```

The measured ratios match the formula to under one per cent, and the last table shows the size of the
error the careless accounting makes: it turns a loss of a factor of two into no change at all.

### 2.3 The control variate coefficient

With $\mathbb{E}g$ known, define $Z_\beta = f - \beta(g - \mathbb{E}g)$. Then
$\mathbb{E}Z_\beta = \mathbb{E}f$ for every $\beta$, so the estimator is unbiased whatever $\beta$ is,
and $\beta$ can be chosen freely.

$$
\operatorname{Var}(Z_\beta) = \operatorname{Var}f - 2\beta\operatorname{Cov}(f,g)
+ \beta^{2}\operatorname{Var}g ,
$$

a quadratic in $\beta$ with positive leading coefficient. Its minimum is at

$$
\boxed{\;\beta^{*} = \frac{\operatorname{Cov}(f,g)}{\operatorname{Var}g}\;}
$$

and substituting back,

$$
\operatorname{Var}(Z_{\beta^{*}}) = \operatorname{Var}f
- \frac{\operatorname{Cov}(f,g)^{2}}{\operatorname{Var}g}
= \operatorname{Var}f\left(1 - \rho^{2}\right) ,
$$

using $\rho^{2} = \operatorname{Cov}(f,g)^{2}/(\operatorname{Var}f\operatorname{Var}g)$. So

$$
\frac{\operatorname{Var}(\text{plain})}{\operatorname{Var}(\text{control})}
= \frac{1}{1-\rho^{2}} .
$$

**Two consequences worth stating.** The sign of $\rho$ is irrelevant, because only $\rho^{2}$
appears; a control that anticorrelates works exactly as well. And $\beta^{*}$ is estimated from the
sample, which introduces a bias of order $1/N$, negligible next to the standard error of order
$N^{-1/2}$ but not exactly zero.

```python
import numpy as np
from nalib import montecarlo as mc

out = mc.a_control_variate_achieves_one_minus_rho_squared()
print(f"{'control':>40}{'rho':>12}{'beta':>10}{'1/(1-rho**2)':>15}{'measured':>12}")
for row in out["rows"]:
    print(f"{row['control']:>40}{row['correlation']:>+12.6f}{row['beta']:>10.4f}"
          f"{row['predicted_variance_ratio']:>15.2f}"
          f"{row['measured_variance_ratio']:>12.2f}")
print(f"\nworst relative miss: {out['worst_relative_miss']:.2e}")

print("\nthe variance really is a quadratic in beta, minimized where the formula says:")
rng = np.random.default_rng(42)
dimension, samples = 4, 200000
f = mc.problems(dimension)["smooth"][0]
points = rng.random((samples, dimension))
values = f(points)
helper = 1.0 + np.sum(points, axis=-1)
best = float(np.cov(values, helper, ddof=1)[0, 1] / np.var(helper, ddof=1))
print(f"{'beta':>12}{'variance':>16}")
for factor in (0.0, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0):
    trial = factor * best
    print(f"{trial:>12.4f}"
          f"{float(np.var(values - trial * (helper - (1.0 + 0.5 * dimension)), ddof=1)):>16.8f}")
print(f"  the smallest is at factor 1.0, that is beta = {best:.4f}")

assert out["the_prediction_holds"]
```

The predicted and measured ratios agree to zero relative error, and the sweep confirms that the
variance is smallest exactly at $\beta^{*}$ and rises on both sides.

### 2.4 The stratified variance

Split the cube into $K$ cells of equal volume $1/K$ and draw $n$ points in each, for $N = Kn$ in
total. Let $\mu_j$ and $\sigma_j^{2}$ be the mean and variance of $f$ inside cell $j$.

The estimator is $\hat I = \frac1K\sum_j \bar f_j$ with $\bar f_j$ the cell average, so

$$
\operatorname{Var}(\hat I) = \frac{1}{K^{2}}\sum_j \frac{\sigma_j^{2}}{n}
= \frac{1}{KN}\sum_j \sigma_j^{2}\cdot\frac{K}{K} = \frac{1}{N}\cdot\frac1K\sum_j\sigma_j^{2}
= \frac{\overline{\sigma^{2}}}{N} ,
$$

with $\overline{\sigma^{2}}$ the average within-cell variance.

**Compare with the unstratified variance.** The law of total variance splits the whole variance into
a within part and a between part:

$$
\sigma^{2} = \underbrace{\frac1K\sum_j\sigma_j^{2}}_{\text{within}}
+ \underbrace{\frac1K\sum_j(\mu_j - \mu)^{2}}_{\text{between}} .
$$

Both terms are non-negative, so $\overline{\sigma^{2}} \le \sigma^{2}$ and

$$
\operatorname{Var}(\text{stratified}) \le \operatorname{Var}(\text{plain}) ,
$$

with equality exactly when every cell has the same mean. **Stratification can never hurt**, which is
not true of the antithetic pairing, and the saving is precisely the between-cell share of the
variance.

```python
import numpy as np
from nalib import montecarlo as mc

out = mc.stratification_pays_where_the_integrand_is_smooth()
print(f"{out['per_axis']} strata per axis in {out['dimension']} dimensions, "
      f"{out['cells']} cells")
print(f"{'integrand':>32}{'variance ratio':>16}")
for row in out["rows"]:
    print(f"{row['integrand']:>32}{row['variance_ratio']:>16.2f}")

print("\nthe split into within and between, measured directly:")
rng = np.random.default_rng(42)
dimension, per_axis, per_cell = 2, 16, 40
for key in ("smooth", "ball"):
    f, exact, name = mc.problems(dimension)[key]
    grid = np.stack(np.meshgrid(*([np.arange(per_axis)] * dimension), indexing="ij"),
                    axis=-1).reshape(-1, dimension) / per_axis
    stack = np.stack([f(grid + rng.random(grid.shape) / per_axis)
                      for _ in range(per_cell)])
    within = float(np.mean(np.var(stack, axis=0, ddof=1)))
    total = float(np.var(stack.ravel(), ddof=1))
    print(f"  {name:>32}: total {total:.6f} = within {within:.6f} + between "
          f"{total - within:.6f}, between share {1.0 - within / total:.4f}")

assert out["it_pays_more_on_the_smooth_one"]
```

The between-cell share is what the technique removes, and it is much larger for the smooth integrand
than for the discontinuous one, which is the whole of the measured difference.

### 2.5 Koksma's inequality and the van der Corput exponent

**The inequality.** For $f$ of bounded variation $V(f)$ on $[0,1]$ and points $x_1,\dots,x_N$,

$$
\left|\frac1N\sum_k f(x_k) - \int_0^1 f\right| \;\le\; V(f)\, D_N^{*} .
$$

**Sketch.** Write $f(x) = f(1) - \int_x^1 df$, substitute into both sides, and exchange the order of
integration and summation. The difference collapses to $\int_0^1 \big(F_N(t) - t\big)\,df(t)$ with
$F_N$ the empirical distribution function, and bounding $|F_N(t) - t|$ by $D_N^{*}$ and
$\int|df| = V(f)$ gives the result.

**What it says about the exponent.** The bound separates completely: one factor depends only on the
integrand and the other only on the points. So the integration error inherits the discrepancy's rate
exactly.

For a random sample the discrepancy is $O(N^{-1/2})$ by the law of the iterated logarithm, giving the
usual Monte Carlo rate. For van der Corput in base $b$ the discrepancy is
$O(\log N / N)$, so the integration error should fall like $N^{-1}$ up to a logarithm.

```python
import numpy as np
from nalib import montecarlo as mc

out = mc.the_discrepancy_is_the_reason()
print(f"{'points':>9}{'van der Corput':>18}{'random':>14}{'ratio':>10}"
      f"{'D times N':>13}{'D times N over log N':>23}")
for n, quasi, random_side in zip(out["counts"], out["quasi"], out["random"]):
    print(f"{n:>9}{quasi:>18.3e}{random_side:>14.3e}{random_side / quasi:>10.2f}"
          f"{quasi * n:>13.4f}{quasi * n / np.log(n):>23.4f}")
print(f"\nfitted exponents: van der Corput {out['quasi_power']:.4f}, "
      f"random {out['random_power']:.4f}")

print("\nand the integration error follows it, as Koksma says it must:")
f, exact, _ = mc.problems(1)["smooth"]
variation = float(np.exp(1.0) - 1.0)         # exp is increasing, so V = f(1) - f(0)
print(f"  the variation of exp on [0,1] is {variation:.6f}")
print(f"{'points':>9}{'error':>14}{'bound V D':>14}{'bound / error':>16}")
for n in out["counts"]:
    points = mc.van_der_corput(n, 2)
    error = abs(float(np.mean(np.exp(points))) - exact)
    bound = variation * mc.star_discrepancy_1d(points)
    print(f"{n:>9}{error:>14.3e}{bound:>14.3e}{bound / error:>16.1f}")

assert out["quasi_is_much_faster"]
```

The measured exponent is $-1.0000$, and the column $D_N^{*}N$ is nearly constant while
$D_N^{*}N/\log N$ falls, so the discrepancy here is closer to $1/N$ than to $\log N/N$: base two van
der Corput is better than the general bound.

The last table is Koksma checked directly. The bound holds at every sample size and it is loose by a
factor of ten to a few hundred, which is typical: the inequality has to cover the worst possible
function of that variation, and $\exp$ is not it.

### 3.1 Importance sampling

Draw from a density $p$ instead of the uniform and reweight:

$$
\int f = \int \frac{f}{p}\,p = \mathbb{E}_p\!\left[\frac{f(X)}{p(X)}\right] .
$$

The variance is $\operatorname{Var}_p(f/p)$, which is **zero** when $p \propto f$. That is the
theoretical optimum and it is unattainable, since normalizing $p$ requires the integral being sought.
What is attainable is $p$ roughly proportional to $f$, and the measurement below shows how much that
is worth.

```python
import math

import numpy as np
from nalib import montecarlo as mc

dimension, samples, repeats = 4, 40000, 24
print(f"{'peak width':>12}{'exact':>12}{'share of the cube with f > 0.01':>34}"
      f"{'plain rms':>13}{'importance rms':>17}{'gain':>16}")
for peak in (0.25, 0.10, 0.05):
    def sharp(x, peak=peak):
        v = np.asarray(x, dtype=float)
        return np.exp(-np.sum((v - 0.5) ** 2, axis=-1) / (2.0 * peak ** 2))

    exact = (peak * math.sqrt(2.0 * math.pi)
             * math.erf(0.5 / (peak * math.sqrt(2.0)))) ** dimension
    plain = np.array([np.mean(sharp(np.random.default_rng(s).random((samples, dimension))))
                      for s in range(repeats)])
    weighted = []
    for s in range(repeats):
        draw = np.random.default_rng(1000 + s)
        points = 0.5 + peak * draw.standard_normal((samples, dimension))
        inside = np.all((points >= 0.0) & (points < 1.0), axis=1)
        density = np.prod(np.exp(-(points - 0.5) ** 2 / (2.0 * peak ** 2))
                          / (peak * math.sqrt(2.0 * math.pi)), axis=-1)
        weighted.append(np.mean(np.where(inside, sharp(points) / density, 0.0)))
    weighted = np.array(weighted)
    plain_rms = float(np.sqrt(np.mean(((plain - exact) / exact) ** 2)))
    weighted_rms = float(np.sqrt(np.mean(((weighted - exact) / exact) ** 2)))
    share = float(np.mean(sharp(np.random.default_rng(0).random((200000, dimension)))
                          > 0.01))
    print(f"{peak:>12.2f}{exact:>12.3e}{share:>34.4f}{plain_rms:>13.3e}"
          f"{weighted_rms:>17.3e}{plain_rms / weighted_rms:>16.3e}")
```

The gain tracks how concentrated the integrand is, and the last row is the theoretical optimum
arriving in practice.

**At a width of $0.25$ the integrand is not concentrated at all**, occupying $94$ per cent of the
cube, and the gain is $1.95$.

**At $0.10$ only $4$ per cent of the cube matters** and the gain is $6900$.

**At $0.05$ only $0.3$ per cent matters, and the gain is $8\times10^{14}$.** At that width the
Gaussian proposal lies entirely inside the cube, so $f/p$ is **exactly constant** and the variance is
exactly zero: every draw gives the same value and the answer is right to rounding. That is
$p \propto f$ achieved by accident, and it shows what the optimum is worth when it is reachable.

The failure mode of the technique is the mirror image. If $p$ has lighter tails than $f$, the ratio
$f/p$ is unbounded, the variance is infinite, and the estimator gives a plausible number with a
meaningless error bar. **A proposal must never decay faster than the integrand**, which is why
practical importance sampling uses heavy tailed proposals even when the target is light tailed.

### 3.2 Latin hypercube sampling

Full stratification needs $k^{d}$ cells, which is impossible past about ten dimensions. Latin
hypercube sampling stratifies **each axis separately**: divide every axis into $N$ intervals and
choose the points so that each interval is used exactly once on each axis, with the assignments
permuted independently.

```python
import numpy as np
from nalib import montecarlo as mc


def latin_hypercube(f, dimension, samples, seed=0):
    """One point per interval on every axis, with the axes permuted independently."""
    rng = np.random.default_rng(int(seed))
    points = np.empty((samples, dimension))
    for axis in range(dimension):
        points[:, axis] = (rng.permutation(samples) + rng.random(samples)) / samples
    return float(np.mean(f(points)))


samples, repeats = 4096, 30
print(f"{'dimension':>11}{'cells full needs':>19}{'plain rms':>13}{'latin rms':>13}"
      f"{'gain':>9}")
for dimension in (2, 5, 10):
    f, exact, _ = mc.problems(dimension)["smooth"]
    plain = np.array([mc.integrate(f, dimension, samples, seed=s)["estimate"]
                      for s in range(repeats)])
    latin = np.array([latin_hypercube(f, dimension, samples, seed=s)
                      for s in range(repeats)])
    plain_rms = float(np.sqrt(np.mean(((plain - exact) / exact) ** 2)))
    latin_rms = float(np.sqrt(np.mean(((latin - exact) / exact) ** 2)))
    print(f"{dimension:>11}{f'16**{dimension} = {16 ** dimension}':>19}"
          f"{plain_rms:>13.4e}{latin_rms:>13.4e}{plain_rms / latin_rms:>9.2f}")

print("\nfull stratification at ten dimensions, for comparison:")
try:
    mc.stratified(mc.problems(10)["smooth"][0], 10, 16, 2)
except ValueError as complaint:
    print(f"  refused: {complaint}")
```

**Latin hypercube gives a gain of $4.48$, $3.73$ and $1.98$ at two, five and ten dimensions**, at a
cost of one permutation per axis and nothing else. Full stratification at sixteen strata per axis
would need $16^{10} = 1.1\times10^{12}$ cells at ten dimensions, and the library refuses to try.

The pattern in the gain is the interesting part. **It shrinks with the dimension**, because Latin
hypercube removes only the variance attributable to the **main effects**, the part of $f$ that is a
sum of one dimensional functions. As the dimension grows, the interaction terms carry a larger share
of the total variance and are untouched.

That is the exact statement of what the method does, and it explains both its popularity in design of
experiments, where main effects dominate by construction, and its modest gain here.

### 3.3 Scrambling a Halton sequence

Plain Halton loses to random sampling by sixteen dimensions, and the reason is that the higher prime
bases need very many points before their digits have cycled. **Scrambling** applies a random
permutation to the digits in each base, which destroys the correlation between axes while preserving
the low discrepancy. **Sobol** replaces the prime bases with base two throughout and a table of
direction numbers, avoiding the problem at its source.

```python
import numpy as np
from scipy.stats import qmc

from nalib import montecarlo as mc

samples, repeats, baseline = 8192, 9, 40
print(f"{'d':>4}{'plain Halton':>15}{'scrambled Halton':>19}{'Sobol':>13}"
      f"{'random rms':>13}{'best over random':>18}")
for dimension in (2, 4, 8, 16, 24):
    f, exact, _ = mc.problems(dimension)["smooth"]
    plain = abs(mc.quasi_integrate(f, dimension, samples)["estimate"] - exact) / exact
    scrambled = float(np.median([
        abs(np.mean(f(qmc.Halton(dimension, scramble=True, seed=s).random(samples))) - exact)
        / exact for s in range(repeats)]))
    sobol = float(np.median([
        abs(np.mean(f(qmc.Sobol(dimension, scramble=True, seed=s).random(samples))) - exact)
        / exact for s in range(repeats)]))
    random_rms = float(np.sqrt(np.mean([
        ((mc.integrate(f, dimension, samples, seed=s)["estimate"] - exact) / exact) ** 2
        for s in range(baseline)])))
    print(f"{dimension:>4}{plain:>15.3e}{scrambled:>19.3e}{sobol:>13.3e}"
          f"{random_rms:>13.3e}{random_rms / min(plain, scrambled, sobol):>18.1f}")
```

**Scrambling repairs plain Halton completely.** At sixteen dimensions plain Halton is $3.3\times10^{-2}$,
**worse** than the random sample's $1.8\times10^{-2}$, while scrambled Halton is $3.0\times10^{-3}$
and still six times better than random. At twenty four dimensions plain Halton is worse than random
by a factor of $2.5$ and scrambled is better by $5.2$.

**Sobol is better still in low dimensions, by a very large margin.** At two dimensions it reaches
$3.1\times10^{-8}$ against plain Halton's $3.4\times10^{-4}$, four orders of magnitude, and against
the random sample's $4.3\times10^{-3}$ it is a factor of $136000$.

**The advantage shrinks with dimension for all of them**, from $136000$ at $d=2$ to $5.2$ at
$d=24$, but the crossover where quasi-random stops being worth using has moved from about eight
dimensions to well past twenty four.

Scrambling also restores something plain Halton cannot give: because the scramble is random, several
scrambles give an **error bar**, which is exercise 5.3.

### 3.4 Sobol against Halton

The table above carries the Sobol column, and it deserves its own reading.

**Sobol is better than either Halton in every dimension tested.** At two dimensions it reaches
$3.1\times10^{-8}$ against scrambled Halton's $6.9\times10^{-5}$, a factor of $2200$, and against
plain Halton's $3.4\times10^{-4}$ a factor of $11000$. At sixteen dimensions it is $1.2\times10^{-3}$
against $3.0\times10^{-3}$.

**Only at twenty four does scrambled Halton overtake it**, $4.9\times10^{-3}$ against
$1.2\times10^{-2}$, and both are still better than random.

The reason Sobol survives high dimensions is structural. Halton uses a different prime base per axis,
and the $j$-th prime needs $p_j$ points before its first digit has cycled once, so at $j = 24$ the
base is $89$ and a few thousand points barely start. **Sobol uses base two on every axis** and
distinguishes them with direction numbers instead, so no axis is intrinsically slower than another.
The cost is that the direction numbers have to be chosen well, and the published tables are the
result of a substantial search.

That is why financial practice uses scrambled Sobol at hundreds of dimensions and plain Halton is a
teaching example.

### 3.5 Three estimates of $\pi$, ranked before running

Three constructions, and the variance of each can be worked out in advance.

**Hit or miss.** Sample the unit square, count the fraction inside the quarter disc, multiply by $4$.
The indicator has mean $\pi/4$ and variance $\tfrac\pi4(1-\tfrac\pi4) = 0.1685$, so the estimate of
$\pi$ has variance $16 \times 0.1685 = 2.697$.

**The quarter circle's height.** Average $4\sqrt{1-x^{2}}$ over $x$ uniform. Its mean is $\pi$ and its
variance is $16(\tfrac23) - \pi^{2} = 0.797$.

**The arctangent integrand.** Average $4/(1+x^{2})$. Its mean is $\pi$ and

$$
\int_0^1\frac{dx}{(1+x^{2})^{2}}
= \left[\frac{x}{2(1+x^{2})} + \frac12\arctan x\right]_0^1 = \frac14 + \frac\pi8 ,
$$

so its variance is $16(\tfrac14 + \tfrac\pi8) - \pi^{2} = 0.4136$.

So the ranking before running is arctangent, then height, then hit or miss, in the ratio
$1 : 1.93 : 6.52$.

```python
import math

import numpy as np

samples, repeats = 200000, 200
predicted = {
    "hit or miss": 16.0 * (math.pi / 4.0) * (1.0 - math.pi / 4.0),
    "the circle's height": 16.0 * (2.0 / 3.0) - math.pi ** 2,
    "the arctangent": 16.0 * (math.pi / 8.0 + 0.25) - math.pi ** 2,
}
estimators = {
    "hit or miss": lambda u: 4.0 * (u[:, 0] ** 2 + u[:, 1] ** 2 <= 1.0),
    "the circle's height": lambda u: 4.0 * np.sqrt(np.maximum(1.0 - u[:, 0] ** 2, 0.0)),
    "the arctangent": lambda u: 4.0 / (1.0 + u[:, 0] ** 2),
}
print(f"{'estimator':>22}{'predicted variance':>21}{'measured':>12}{'ratio':>9}"
      f"{'rms error':>13}")
for name in ("the arctangent", "the circle's height", "hit or miss"):
    values = []
    for s in range(repeats):
        draws = np.random.default_rng(s).random((samples, 2))
        values.append(float(np.mean(estimators[name](draws))))
    values = np.array(values)
    measured = float(np.var(values, ddof=1)) * samples
    print(f"{name:>22}{predicted[name]:>21.6f}{measured:>12.6f}"
          f"{measured / predicted[name]:>9.4f}"
          f"{float(np.sqrt(np.mean((values - math.pi) ** 2))):>13.3e}")
print(f"\nthe predicted ranking, worst over best: "
      f"{predicted['hit or miss'] / predicted['the arctangent']:.2f}")
```

Every predicted variance is confirmed to within the sampling error of a variance estimated from
$200$ runs, which is a few per cent, and the ranking holds exactly. The lesson is that **the variance
is computable in advance**, so the choice among estimators of the same quantity is a calculation
rather than an experiment.

Hit or miss is the version everybody meets first and it is the worst of the three by a factor of
$6.5$, because it throws away all the information in **where** the point landed and keeps only
whether it was inside. The arctangent version is best because its integrand varies least: it runs
from $4$ to $2$ across the interval, where the circle's height runs from $4$ to $0$.

### 4.1 Coverage against skewness

The central limit theorem is a limit. How many samples it needs before the error bar has its intended
coverage depends on how far the integrand is from normal, and the skewness is the leading term in the
Edgeworth expansion that measures it.

```python
import math

import numpy as np

budgets = (20, 100, 500, 2500, 12500)
trials = 1200
cases = (("uniform, sum x", lambda u: u.sum(axis=-1)),
         ("moderate, exp(3x)", lambda u: np.exp(3.0 * u).prod(axis=-1)),
         ("heavy, 1/sqrt(x)", lambda u: (1.0 / np.sqrt(np.maximum(u, 1e-300))).prod(axis=-1)))
print(f"{'integrand':>22}{'skewness':>11}"
      + "".join(f"{f'N={n}':>9}" for n in budgets))
for name, f in cases:
    big = f(np.random.default_rng(0).random((400000, 1)))
    skew = float(np.mean((big - big.mean()) ** 3) / big.std() ** 3)
    truth = float(np.mean(f(np.random.default_rng(9).random((4000000, 1)))))
    row = []
    for n in budgets:
        inside = 0
        for s in range(trials):
            values = f(np.random.default_rng(10000 + s).random((n, 1)))
            bar = float(np.std(values, ddof=1)) / math.sqrt(n)
            inside += abs(float(np.mean(values)) - truth) < bar
        row.append(inside / trials)
    print(f"{name:>22}{skew:>11.3f}" + "".join(f"{v:>9.3f}" for v in row))
print(f"{'a normal predicts':>22}{0.0:>11.3f}"
      + "".join(f"{math.erf(1.0 / math.sqrt(2.0)):>9.3f}" for _ in budgets))
```

**A symmetric integrand needs no samples at all**: coverage is $0.682$ at $N = 20$ against a
predicted $0.683$.

**Moderate skewness costs a little**: $0.659$ at $N = 20$, recovering to $0.683$ by $N = 12500$.

**Heavy skewness costs a great deal.** At a skewness of $118$ the coverage is $0.557$ at $N = 20$, so
one run in twenty five more than the nominal rate falls outside its own bar, and even at $N = 12500$
it is $0.671$ and still climbing.

The rule of thumb from the Edgeworth expansion is that the coverage error is of order
$\gamma/\sqrt N$ with $\gamma$ the skewness, so reaching a fixed coverage error needs
$N \propto \gamma^{2}$. At $\gamma = 118$ that is $14000$ times the samples the symmetric case needs,
which is what the table shows.

The practical consequence: **for a heavy tailed integrand the reported bar is optimistic, and the
first thing to measure is the skewness**, not the mean.

### 4.2 Against a product Gauss rule

Simpson is fourth order. Gauss-Legendre with $n$ nodes is exact for polynomials of degree $2n-1$, so
for a smooth integrand it is far better. Does that move the crossover?

```python
import numpy as np
from nalib import montecarlo as mc

budget, repeats = 100000, 9
print(f"{'d':>4}{'nodes per axis':>16}{'points':>10}{'Gauss error':>14}"
      f"{'MC error':>13}{'MC wins':>9}   integrand")
for dimension in (2, 3, 4, 5, 6):
    for key in ("smooth", "ball"):
        f, exact, name = mc.problems(dimension)[key]
        per_axis = max(int(round(budget ** (1.0 / dimension))), 3)
        used = per_axis ** dimension
        nodes, weights = np.polynomial.legendre.leggauss(per_axis)
        nodes = 0.5 * (nodes + 1.0)
        weights = 0.5 * weights
        grid = np.stack(np.meshgrid(*([nodes] * dimension), indexing="ij"),
                        axis=-1).reshape(-1, dimension)
        weight = np.ones(tuple([per_axis] * dimension))
        for axis in range(dimension):
            shape = [1] * dimension
            shape[axis] = per_axis
            weight = weight * weights.reshape(shape)
        gauss = abs(float(np.sum(weight.reshape(-1) * f(grid))) - exact) / abs(exact)
        sampled = np.sqrt(np.mean([
            ((mc.integrate(f, dimension, used, seed=s)["estimate"] - exact) / exact) ** 2
            for s in range(repeats)]))
        print(f"{dimension:>4}{per_axis:>16}{used:>10}{gauss:>14.3e}"
              f"{sampled:>13.3e}{str(sampled < gauss):>9}   {name}")
```

**On the separable integrand Gauss is a different order of thing.** Its errors are $10^{-15}$ to
$10^{-16}$, at rounding, where Simpson's ranged from $10^{-12}$ to $10^{-4}$. That is the spectral
accuracy of Gauss quadrature on an entire function, and Monte Carlo is nowhere near it at any
dimension.

**On the discontinuous integrand nothing changes.** The crossover is at $d = 4$, exactly where it was
with Simpson, and the Gauss errors are the same size as Simpson's.

That is the answer, and it is worth stating carefully. **The order of the quadrature rule is
irrelevant when the integrand is not smooth.** The error there comes from the cells the discontinuity
crosses, where the integrand is not approximated by any polynomial, and there are
$O(N^{(d-1)/d})$ of them whatever weights are used. Improving the rule from fourth order to spectral
buys nothing at all.

So the crossover dimension is a property of the **integrand's smoothness**, not of the rule, and the
usual argument that names Simpson and derives $d = 8$ has the wrong variable in it.

### 4.3 Do the savings multiply?

```python
import math

import numpy as np
from nalib import montecarlo as mc

dimension, samples = 4, 400000
f, exact, _ = mc.problems(dimension)["smooth"]
control = lambda x: np.prod(1.0 + x + 0.5 * x * x, axis=-1)
control_mean = (1.0 + 0.5 + 1.0 / 6.0) ** dimension

plain = mc.integrate(f, dimension, samples)
folded = mc.antithetic(f, dimension, samples)
guided = mc.control_variate(f, control, control_mean, dimension, samples)

rng = np.random.default_rng(42)
points = rng.random((samples // 2, dimension))
paired = 0.5 * (f(points) + f(1.0 - points))
paired_control = 0.5 * (control(points) + control(1.0 - points))
beta = float(np.cov(paired, paired_control, ddof=1)[0, 1]
             / np.var(paired_control, ddof=1))
adjusted = paired - beta * (paired_control - control_mean)
both = float(np.std(adjusted, ddof=1)) / math.sqrt(adjusted.size)

print(f"{'estimator':>16}{'standard error':>18}{'variance ratio':>17}")
for label, bar in (("plain", plain["standard_error"]),
                   ("antithetic", folded["standard_error"]),
                   ("control", guided["standard_error"]),
                   ("both together", both)):
    print(f"{label:>16}{bar:>18.4e}{(plain['standard_error'] / bar) ** 2:>17.2f}")
separate = ((plain["standard_error"] / folded["standard_error"]) ** 2
            * (plain["standard_error"] / guided["standard_error"]) ** 2)
print(f"\nif the savings multiplied, the combined ratio would be {separate:.2f}")
print(f"it is {(plain['standard_error'] / both) ** 2:.2f}, which is "
      f"{(plain['standard_error'] / both) ** 2 / separate:.3f} of that")
```

**The savings do not multiply.** Separately the techniques give $4.19$ and $333.5$, whose product is
$1398$. Together they give $489$, about a third of it.

The reason is that they are exploiting the same structure. The control variate is a smooth monotone
function correlating with $f$ at $0.9985$; once it has been subtracted, the residual is what the
control could not explain, and that residual is much less monotone than $f$ was. The antithetic
pairing then has far less to work with: on its own it gains $4.19$, and applied after the control it
gains only $489/333.5 = 1.47$.

The general principle is that **variance reduction techniques compose sub-multiplicatively**, because
each removes part of the same total variance. The exception is techniques that attack genuinely
independent components: stratifying the axes and controlling the smooth part do combine well,
because one removes between-cell variance and the other removes the systematic trend.

The practical reading is to apply the strongest technique first and measure what the next one adds,
rather than assuming the factors compound.

### 5.1 Why $1/\sqrt N$ cannot be improved

**The claim.** No estimator based on $N$ independent uniform samples, unbiased for every integrand in
a class containing the indicators, can have worst case error below $c\,N^{-1/2}$.

**The argument.** Take the class of integrands $f_A = \mathbb{1}_A$ for measurable $A \subseteq
[0,1]^d$, whose integral is $\operatorname{vol}(A)$. Given $N$ independent samples, an estimator sees
only the values $f_A(U_1),\dots,f_A(U_N)$, which is a sequence of $N$ Bernoulli draws with parameter
$p = \operatorname{vol}(A)$.

Estimating a Bernoulli parameter from $N$ independent draws is a classical problem, and the
Cramér-Rao bound gives, for any unbiased estimator,

$$
\operatorname{Var}(\hat p) \;\ge\; \frac{1}{N\,I(p)} = \frac{p(1-p)}{N} ,
$$

with $I(p) = 1/(p(1-p))$ the Fisher information. At $p = 1/2$ that is $1/(4N)$, so the standard error
is at least $\tfrac12 N^{-1/2}$ and no estimator can do better.

**What the argument uses.** Independence, which makes the information add; unbiasedness, which
Cramér-Rao requires; and a class rich enough to contain a hard instance. Each is a place where the
bound can be escaped, and the three escapes are the three families of methods that beat it.

**How quasi-random methods get around it.** They abandon **independence**. A low discrepancy sequence
is deterministic, its points are strongly negatively correlated by construction, and the information
argument does not apply because there is no sampling distribution to take the information of. What
replaces the bound is Koksma's inequality, which gives $O((\log N)^{d}/N)$ but needs bounded variation
and offers no confidence interval.

The other two escapes are worth naming. **Multilevel methods** abandon the single sample size,
spending most of their samples on a cheap approximation and few on the expensive one, which beats the
bound in **cost** without beating it in $N$. And **adaptive or biased estimators** abandon
unbiasedness, trading a small bias for a much smaller variance, which is what regularized and
shrinkage estimators do.

### 5.2 When the variance is infinite

**A construction.** On $[0,1]$ take

$$
f(x) = x^{-\alpha} , \qquad \tfrac12 < \alpha < 1 .
$$

Then

$$
\int_0^1 x^{-\alpha}dx = \frac{1}{1-\alpha} < \infty ,
\qquad
\int_0^1 x^{-2\alpha}dx = \infty \text{ when } \alpha \ge \tfrac12 .
$$

So the integral exists and the variance does not. At $\alpha = 0.7$ the integral is $3.3333$ exactly.

```python
import math

import numpy as np

alpha, trials, seeds = 0.7, 400, 60
exact = 1.0 / (1.0 - alpha)
print(f"the integral of x**-{alpha} on [0,1] is {exact:.6f}, and its variance is infinite")
print(f"{'samples':>10}{'estimate':>14}{'reported bar':>15}{'true error':>14}"
      f"{'error / bar':>14}")
for n in (1000, 10000, 100000, 1000000):
    covered = 0
    ratios = []
    for s in range(trials):
        draws = np.random.default_rng(s).random(n)
        values = draws ** -alpha
        estimate = float(np.mean(values))
        bar = float(np.std(values, ddof=1)) / math.sqrt(n)
        covered += abs(estimate - exact) < bar
        ratios.append(abs(estimate - exact) / bar)
    draws = np.random.default_rng(0).random(n)
    values = draws ** -alpha
    estimate = float(np.mean(values))
    bar = float(np.std(values, ddof=1)) / math.sqrt(n)
    print(f"{n:>10}{estimate:>14.6f}{bar:>15.6f}{abs(estimate - exact):>14.6f}"
          f"{abs(estimate - exact) / bar:>14.4f}")
print(f"\ncoverage over {trials} runs at the largest size: {covered / trials:.3f}, "
      f"against a nominal 0.683")

print("\nand the reported bar does not settle, because s itself has no limit:")
for n in (1000, 10000, 100000, 1000000):
    bars = [float(np.std(np.random.default_rng(s).random(n) ** -alpha, ddof=1))
            for s in range(seeds)]
    print(f"  N = {n:>8}: sample standard deviation ranges "
          f"{min(bars):.3f} to {max(bars):.3f}, a spread of "
          f"{max(bars) / min(bars):.2f}")
```

**The estimator still converges**, because the law of large numbers needs only a finite mean. What
fails is everything built on the variance.

**The reported bar is meaningless.** The sample standard deviation $s$ has no limit to converge to,
and the measurement shows it: across sixty seeds at a million samples it ranges from $17$ to $156$, a
spread of a factor of $9$, and at a thousand samples from $3.2$ to $404$, a factor of $128$. The
spread does not shrink in any orderly way with $N$, because whether a run has yet met a large outlier
is itself the dominant random variable.

**And the coverage is wrong**, below the nominal $68.3$ per cent, because the error distribution is a
stable law with tails far heavier than a normal's rather than a normal.

**The two standard repairs.**

*Transform the integrand.* Substitute $x = t^{1/(1-\alpha)}$, which turns $\int_0^1 x^{-\alpha}dx$
into an integral of a bounded function with finite variance. This is exact and it is always the first
thing to try, because the singularity is usually known.

*Importance sample.* Draw from a density with the same singularity, so that $f/p$ is bounded. This is
exercise 3.1's optimum applied to the tail rather than to the peak, and it works when the singularity
is known well enough to mimic but not well enough to remove.

A third, weaker repair is to **truncate**: discard or cap values above a threshold. That restores a
finite variance and introduces a bias, and the bias is usually harder to quantify than the problem it
solves.

### 5.3 Randomized quasi-Monte Carlo

**The problem with plain Halton.** Its error obeys Koksma's inequality, which needs $V(f)$ and is
loose by orders of magnitude, as exercise 2.5 measured. There is no other bound, and there is no
sampling distribution, so there is no confidence interval. A practitioner gets a number with no
indication of how far off it might be.

**The repair.** Randomize the sequence in a way that preserves its low discrepancy:

- **Random shift.** Add a single uniform vector $\Delta$ modulo one to every point. The shifted set
  has the same discrepancy up to a constant, and each point is individually uniform.
- **Digital scrambling.** Permute the digits of each coordinate in its base, with independent random
  permutations at each digit position. This is Owen's scrambling and it is what exercise 3.3 used.

**Why it recovers an error bar.** After randomization the estimator $\hat I_j$ from scramble $j$ is
**unbiased**, because each point is uniformly distributed. So running $R$ independent scrambles gives
$R$ independent unbiased estimates, and

$$
\hat I = \frac1R\sum_j \hat I_j , \qquad
\text{standard error} = \frac{1}{\sqrt R}\,\text{sd}(\hat I_1,\dots,\hat I_R) .
$$

That is an ordinary Monte Carlo error bar computed over scrambles rather than over points, and it is
valid for exactly the reason section 1's was.

```python
import math

import numpy as np
from scipy.stats import qmc

from nalib import montecarlo as mc

dimension, samples = 6, 4096
f, exact, _ = mc.problems(dimension)["smooth"]
print(f"{'scrambles':>11}{'estimate':>14}{'reported bar':>15}{'true error':>14}"
      f"{'inside':>9}")
for replications in (4, 8, 16, 32):
    estimates = np.array([
        float(np.mean(f(qmc.Sobol(dimension, scramble=True, seed=s).random(samples))))
        for s in range(replications)])
    estimate = float(np.mean(estimates))
    bar = float(np.std(estimates, ddof=1)) / math.sqrt(replications)
    print(f"{replications:>11}{estimate:>14.8f}{bar:>15.3e}"
          f"{abs(estimate - exact):>14.3e}"
          f"{str(abs(estimate - exact) < 2.0 * bar):>9}")

plain = abs(mc.quasi_integrate(f, dimension, samples * 32)["estimate"] - exact)
print(f"\nplain Halton with the same total points: error {plain:.3e}, and no bar at all")
```

The bar is honest and the answer is inside it, at every replication count.

**What it costs.** The total budget is split between points and scrambles, and the two pull against
each other: the accuracy improves with the points at close to $N^{-1}$, and the **bar** improves with
the scrambles at only $R^{-1/2}$. So a useful bar costs a real fraction of the budget, and the usual
compromise is a handful of scrambles, enough for a rough bar and not enough for a precise one.

There is a second benefit worth naming. Scrambled nets are provably **no worse** than plain Monte
Carlo, so randomization removes the failure mode of exercise 3.3, where plain Halton at sixteen
dimensions lost to random sampling. The bar and the safety net come together.

---

## Lesson 93, Brownian Motion and Stochastic Differential Equations

### 1.1 The two defining properties

A standard Brownian motion $W$ on $[0,\infty)$ satisfies:

**Independent increments.** For $s < t \le u < v$, the increments $W(t)-W(s)$ and $W(v)-W(u)$ are
independent.

**Normal increments with variance equal to the elapsed time.** $W(t)-W(s) \sim N(0, t-s)$.

Together with $W(0)=0$ and continuity of the paths, these determine the process completely.

Two things follow at once and are worth separating from the definition.

The **variance is the elapsed time**, so the typical displacement over a time $h$ is $\sqrt h$, not
$h$. That single scaling is the origin of everything in this lesson.

The increments are **stationary**: the distribution of $W(t+h)-W(t)$ does not depend on $t$. The
definition does not state this separately because it follows from the second property.

The lesson measures both: variance $1$ at time $1$ and $0.5$ at time $\tfrac12$, with lag one
correlations below $0.0021$.

### 1.2 Why the path has no derivative

The difference quotient over a step $h$ is

$$
\frac{W(t+h) - W(t)}{h} \sim \frac{N(0, h)}{h} = N\!\left(0, \frac1h\right) ,
$$

whose standard deviation is $h^{-1/2}$. As $h \to 0$ that diverges, so the difference quotient has
no limit: it does not settle down, it spreads out without bound.

The scaling is the whole argument. A differentiable function has increments of order $h$, so its
quotient is $O(1)$. Brownian motion has increments of order $\sqrt h$, so its quotient is
$O(h^{-1/2})$.

Two stronger statements follow from the same scaling. The path has **unbounded variation** on every
interval: summing $|{\Delta W}| \approx \sqrt h$ over $1/h$ steps gives $h^{-1/2} \to \infty$. And
its **quadratic variation** is finite and non-zero, summing $(\Delta W)^2 \approx h$ over $1/h$ steps
to give the elapsed time, which section 2 measures at $1.006$.

Those two together are exactly what makes an ordinary Riemann-Stieltjes integral against $dW$
impossible and an Ito integral necessary.

### 1.3 Strong and weak order

**Strong order $p$** means

$$
\mathbb{E}\big|X_N - X(T)\big| = O(h^{p}) ,
$$

where $X(T)$ is the exact solution driven by the **same** Brownian path as the numerical one. It
measures whether the computed **trajectory** is right.

**Weak order $q$** means

$$
\big|\mathbb{E}g(X_N) - \mathbb{E}g(X(T))\big| = O(h^{q})
$$

for smooth $g$. It measures whether the computed **distribution** is right, and says nothing at all
about individual paths.

**Questions that need the strong order.** Anything where one trajectory is the answer: fitting a
model to an observed path, computing a pathwise sensitivity, or a multilevel estimator whose whole
mechanism is the correlation between coarse and fine paths on the same noise. Exercise 5.1 develops
the last.

**Questions that need only the weak order.** Anything that is an expectation: an option price, a
reaction rate, an expected first passage time, a stationary distribution. This is most of what SDEs
are solved for, and it is why exercise 5's answer about Milstein matters.

The two are genuinely different numbers, and the lesson measures $0.4974$ and $0.9950$ for the same
method on the same runs.

### 1.4 What Milstein adds

$$
X_{k+1} = X_k + a\,h + b\,\Delta W + \underbrace{\tfrac12\,b\,b'\,\big(\Delta W^{2} - h\big)}_{\text{the extra term}} .
$$

It is the next term of the Ito-Taylor expansion, and it costs one line plus the derivative $b'$ of
the diffusion coefficient.

**When it is worth adding.** When the strong order matters, and only then. The lesson measures a
factor of $31$ in the strong error at the finest step, and a weak gain of $0.970$, which is no gain.

**When it is not worth adding.** Three cases.

*When only an expectation is wanted*, which is the common case.

*When the noise is additive.* Then $b' = 0$, the term vanishes identically, and the two methods are
the same method. The lesson measures a difference of exactly $0.0$.

*When $b'$ is unavailable or expensive.* Exercise 3.1 gives a derivative free method with the same
strong order.

### 1.5 Why the mean and the median differ

For $dX = \mu X\,dt + \sigma X\,dW$, Ito's lemma applied to $\log X$ gives

$$
d(\log X) = \left(\mu - \tfrac{\sigma^{2}}{2}\right)dt + \sigma\,dW ,
$$

where the $-\sigma^{2}/2$ is the Ito correction. So $\log X(t)$ is normal with mean
$(\mu - \sigma^{2}/2)t$ and variance $\sigma^{2}t$, making $X(t)$ **lognormal**.

A lognormal is skewed, and for it

$$
\text{median} = e^{(\mu - \sigma^{2}/2)t} , \qquad
\text{mean} = e^{\mu t} , \qquad
\frac{\text{mean}}{\text{median}} = e^{\sigma^{2}t/2} .
$$

**The median is what the typical path does. The mean is carried by a few large outcomes.** With
$\mu = 0.1$ and $\sigma = 0.5$ over two years the lesson measures a mean that has risen $22$ per cent
and a median that has **fallen** $5$ per cent: most paths lose money and the average gains.

Ordinary calculus applied to $d(\log X)$ would omit the $-\sigma^{2}/2$ and therefore compute the
median while calling it the mean. That is a standard way to be wrong about a stochastic model, and
the gap is exactly the term Ito's rule adds.

### 2.1 The quadratic variation

**Claim.** For a partition of $[0,T]$ into $n$ equal steps,

$$
Q_n = \sum_{k=0}^{n-1}\big(W(t_{k+1}) - W(t_k)\big)^{2} \longrightarrow T
$$

in $L^{2}$, hence in probability, as $n \to \infty$.

**Proof.** Write $\Delta_k = W(t_{k+1}) - W(t_k)$, which are independent $N(0, h)$ with $h = T/n$.

*The mean is exactly $T$.* $\mathbb{E}[\Delta_k^{2}] = h$, so
$\mathbb{E}[Q_n] = nh = T$ for **every** $n$, not merely in the limit.

*The variance goes to zero.* Since the $\Delta_k$ are independent, the variances add:

$$
\operatorname{Var}(Q_n) = \sum_k \operatorname{Var}(\Delta_k^{2}) = n\big(\mathbb{E}\Delta^{4}
- (\mathbb{E}\Delta^{2})^{2}\big) = n\big(3h^{2} - h^{2}\big) = 2nh^{2} = \frac{2T^{2}}{n} ,
$$

using $\mathbb{E}[N(0,h)^{4}] = 3h^{2}$. So $\operatorname{Var}(Q_n) \to 0$ and
$\mathbb{E}[(Q_n - T)^{2}] = 2T^{2}/n \to 0$. $\square$

**The contrast with a smooth path.** If $f$ is differentiable with $|f'| \le M$ then
$|\Delta_k| \le Mh$, so

$$
\sum_k \Delta_k^{2} \le n M^{2}h^{2} = \frac{M^{2}T^{2}}{n} \longrightarrow 0 ,
$$

falling like $1/n$. That is the difference: one limit is $T$ and the other is $0$, and the reason is
entirely the $\sqrt h$ against $h$ scaling.

```python
import numpy as np
from nalib import sde

out = sde.the_quadratic_variation_is_the_time()
print(f"{'steps':>8}{'Brownian':>14}{'error':>12}{'2T**2/n predicted sd':>23}"
      f"{'smooth path':>15}{'times n':>12}")
for row in out["rows"]:
    predicted = np.sqrt(2.0 * out["horizon"] ** 2 / row["steps"])
    print(f"{row['steps']:>8}{row['brownian']:>14.6f}{row['brownian_error']:>12.2e}"
          f"{predicted:>23.2e}{row['smooth']:>15.3e}"
          f"{row['smooth'] * row['steps']:>12.4f}")

print(f"\nthe smooth path's sum falls like n to the {out['smooth_power']:.4f}")
print(f"the Brownian one reaches {out['rows'][-1]['brownian']:.6f} against "
      f"a horizon of {out['horizon']:g}")

print("\nthe mean is exactly T at every n, which the variance formula also predicts:")
draws = 4000
for steps in (16, 64, 256):
    sums = [sde.quadratic_variation(sde.brownian(steps, paths=1, seed=s)["values"][0],
                                    sde.brownian(steps, paths=1, seed=s)["times"])
            for s in range(draws)]
    print(f"  n = {steps:>4}: mean {float(np.mean(sums)):.6f}, sd "
          f"{float(np.std(sums, ddof=1)):.6f}, predicted sd "
          f"{np.sqrt(2.0 / steps):.6f}")

assert out["brownian_reaches_the_horizon"]
```

The mean is $1$ at every step count and the standard deviation matches $\sqrt{2/n}$ to three digits,
which is the proof confirmed term by term. The smooth path's sum times $n$ is constant, which is the
$1/n$ rate.

### 2.2 Ito's lemma

**Setup.** Expand $f(W(t+h))$ about $W(t)$ to second order:

$$
f(W + \Delta W) = f(W) + f'(W)\,\Delta W + \tfrac12 f''(W)\,(\Delta W)^{2}
+ O\big(|\Delta W|^{3}\big) .
$$

**Sum over a partition.** With $n$ steps of size $h$,

$$
f(W(T)) - f(W(0)) = \sum_k f'(W_k)\,\Delta W_k
+ \tfrac12\sum_k f''(W_k)\,(\Delta W_k)^{2} + \sum_k O\big(|\Delta W_k|^{3}\big) .
$$

**Take each term in turn.**

The first sum is the Ito integral $\int_0^T f'(W)\,dW$ by definition, since the integrand is
evaluated at the **left** endpoint of each step.

The second sum is where the new term appears. By 2.1, $(\Delta W_k)^{2}$ behaves like $h$ rather than
like a mean zero quantity of size $h$: the sum $\sum_k g(W_k)(\Delta W_k)^{2}$ converges to
$\int_0^T g(W)\,dt$, not to zero. So the second sum tends to $\tfrac12\int_0^T f''(W)\,dt$.

The third sum vanishes: $\mathbb{E}|\Delta W|^{3} = O(h^{3/2})$ and there are $1/h$ terms, giving
$O(h^{1/2}) \to 0$.

**Therefore**

$$
\boxed{\;f(W(T)) - f(W(0)) = \int_0^T f'(W)\,dW + \tfrac12\int_0^T f''(W)\,dt\;}
$$

which in differential form is $df = f'\,dW + \tfrac12 f''\,dt$, the rule $(dW)^{2} = dt$ applied
mechanically.

```python
import math

import numpy as np
from nalib import sde

steps, paths, horizon = 20000, 4000, 1.0
path = sde.brownian(steps, horizon=horizon, paths=paths, seed=42)
values, increments = path["values"], path["increments"]
h = path["step"]

for name, f, fp, fpp in (("f(w) = w**2", lambda w: w * w, lambda w: 2.0 * w,
                          lambda w: 2.0 * np.ones_like(w)),
                         ("f(w) = exp(w)", np.exp, np.exp, np.exp),
                         ("f(w) = sin(w)", np.sin, np.cos, lambda w: -np.sin(w))):
    left = values[:, :-1]
    change = f(values[:, -1]) - f(values[:, 0])
    ito = np.sum(fp(left) * increments, axis=1)
    correction = 0.5 * np.sum(fpp(left), axis=1) * h
    with_term = float(np.mean(np.abs(change - ito - correction)))
    without = float(np.mean(np.abs(change - ito)))
    print(f"{name:>16}: with the Ito term {with_term:.3e}, without it {without:.3e}, "
          f"ratio {without / with_term:>8.1f}")
```

For every $f$ the identity holds with the correction and fails without it, by factors of tens to
hundreds. That is the extra term measured directly rather than argued for.

### 2.3 The exact solution of geometric Brownian motion

Apply Ito's lemma in its general form to $Y = \log X$, where $dX = \mu X\,dt + \sigma X\,dW$:

$$
dY = \frac{\partial Y}{\partial x}\,dX + \tfrac12\frac{\partial^{2}Y}{\partial x^{2}}\,(dX)^{2} .
$$

**Compute each piece.** $\partial Y/\partial x = 1/X$ and $\partial^{2}Y/\partial x^{2} = -1/X^{2}$.
For $(dX)^{2}$, expand and apply $(dt)^{2} = 0$, $dt\,dW = 0$, $(dW)^{2} = dt$:

$$
(dX)^{2} = \big(\mu X\,dt + \sigma X\,dW\big)^{2} = \sigma^{2}X^{2}\,dt .
$$

**Substitute.**

$$
dY = \frac{1}{X}\big(\mu X\,dt + \sigma X\,dW\big)
- \frac{1}{2X^{2}}\,\sigma^{2}X^{2}\,dt
= \left(\mu - \frac{\sigma^{2}}{2}\right)dt + \sigma\,dW .
$$

**The right side has no $X$ in it**, so it integrates directly:

$$
\log X(t) - \log X(0) = \left(\mu - \frac{\sigma^{2}}{2}\right)t + \sigma W(t) ,
$$

giving

$$
\boxed{\;X(t) = X(0)\exp\!\left(\left(\mu - \frac{\sigma^{2}}{2}\right)t + \sigma W(t)\right)\;}
$$

**The $-\sigma^{2}/2$ came entirely from the second derivative term**, and it is the difference
between the mean and the median of 1.5. Ordinary calculus would give $\mu t$ in the exponent and the
wrong answer.

```python
import math

import numpy as np
from nalib import sde

problem = sde.geometric_brownian(drift=0.1, volatility=0.5)
horizon, paths = 2.0, 400000
rng = np.random.default_rng(42)
walk = math.sqrt(horizon) * rng.standard_normal(paths)
values = problem["exact"](horizon, walk)

print(f"{'quantity':>14}{'measured':>14}{'predicted':>14}{'ratio':>10}")
print(f"{'mean':>14}{float(np.mean(values)):>14.6f}"
      f"{problem['mean'](horizon):>14.6f}"
      f"{float(np.mean(values)) / problem['mean'](horizon):>10.5f}")
print(f"{'median':>14}{float(np.median(values)):>14.6f}"
      f"{problem['median'](horizon):>14.6f}"
      f"{float(np.median(values)) / problem['median'](horizon):>10.5f}")
print(f"{'variance':>14}{float(np.var(values, ddof=1)):>14.6f}"
      f"{problem['variance'](horizon):>14.6f}"
      f"{float(np.var(values, ddof=1)) / problem['variance'](horizon):>10.5f}")

print(f"\nthe naive exponent would be mu t = "
      f"{problem['drift'] * horizon:.4f}")
print(f"the correct one is (mu - sigma**2/2) t = "
      f"{(problem['drift'] - 0.5 * problem['volatility'] ** 2) * horizon:.4f}")
print(f"the difference is the Ito term, sigma**2 t / 2 = "
      f"{0.5 * problem['volatility'] ** 2 * horizon:.4f}")

out = sde.the_ito_correction_is_measurable()
assert out["the_correction_matches"]
```

All three moments of the closed form match to four digits, and the naive and correct exponents differ
by exactly the Ito term.

### 2.4 The Milstein term

Start from the integral form over one step:

$$
X_{k+1} = X_k + \int_{t_k}^{t_{k+1}}a(X_s)\,ds + \int_{t_k}^{t_{k+1}}b(X_s)\,dW_s .
$$

**Euler-Maruyama** freezes both integrands at $X_k$, giving $a\,h + b\,\Delta W$. The error in the
$ds$ integral is $O(h^{2})$, which is fine. The error in the $dW$ integral is $O(h)$, which is not,
and that is what limits the strong order.

**So expand the diffusion integrand.** By Ito's lemma applied to $b(X_s)$,

$$
b(X_s) = b(X_k) + \int_{t_k}^{s}b'(X_r)\,dX_r + \tfrac12\int_{t_k}^{s}b''(X_r)\,(dX_r)^{2} ,
$$

and keeping only the leading term of $dX_r \approx b\,dW_r$,

$$
b(X_s) \approx b(X_k) + b(X_k)b'(X_k)\big(W_s - W_{t_k}\big) .
$$

**Substitute back.**

$$
\int_{t_k}^{t_{k+1}}b(X_s)\,dW_s \approx b\,\Delta W
+ b\,b'\int_{t_k}^{t_{k+1}}\big(W_s - W_{t_k}\big)\,dW_s .
$$

**Evaluate the double integral.** By Ito's lemma applied to $f(w) = w^{2}$ over the step,

$$
\Delta W^{2} = 2\int_{t_k}^{t_{k+1}}(W_s - W_{t_k})\,dW_s + h ,
$$

so the integral is $\tfrac12(\Delta W^{2} - h)$. Therefore

$$
\boxed{\;X_{k+1} = X_k + a\,h + b\,\Delta W + \tfrac12\,b\,b'\,\big(\Delta W^{2} - h\big)\;}
$$

**Its mean is zero**, since $\mathbb{E}[\Delta W^{2}] = h$, which is why adding it does not change
the weak order.

```python
import math

import numpy as np
from nalib import sde

steps, paths = 4000, 20000
h = 1.0 / steps
rng = np.random.default_rng(42)
increments = math.sqrt(h) * rng.standard_normal((paths, steps))

print("the double integral against the closed form (dW**2 - h)/2:")
walk = np.concatenate([np.zeros((paths, 1)), np.cumsum(increments, axis=1)], axis=1)
by_sum = np.sum(walk[:, :-1] * increments, axis=1)
closed = 0.5 * (walk[:, -1] ** 2 - 1.0)
print(f"  worst difference {float(np.max(np.abs(by_sum - closed))):.3e}")

print(f"\nthe extra term has mean zero, which is why the weak order does not move:")
term = 0.5 * (increments[:, 0] ** 2 - h)
print(f"  mean {float(np.mean(term)):+.3e}, and the sampling bar is "
      f"{float(np.std(term, ddof=1)) / math.sqrt(paths):.3e}")
print(f"  its standard deviation is {float(np.std(term, ddof=1)):.3e}, "
      f"against h / sqrt(2) = {h / math.sqrt(2.0):.3e}")

out = sde.milstein_doubles_the_strong_order_only()
assert out["milstein_is_strong_order_one"]
```

The sum and the closed form agree to rounding, the term's mean is at the sampling bar, and its
standard deviation is $h/\sqrt2$ as the normal fourth moment predicts.

### 2.5 Why Euler-Maruyama is weak order one

The term Euler-Maruyama drops, from 2.4, is

$$
E_k = \tfrac12\,b\,b'\,\big(\Delta W_k^{2} - h\big) .
$$

**Its mean is exactly zero**, because $\mathbb{E}[\Delta W_k^{2}] = h$ and $b, b'$ are evaluated at
$X_k$, which is independent of $\Delta W_k$.

**So it cannot contribute at first order to any expectation.** Summing over $n = T/h$ steps, the
strong error accumulates the terms themselves, each of size $O(h)$, giving
$\mathbb{E}|{\sum E_k}| = O(\sqrt{n}\cdot h) = O(\sqrt h)$: the square root comes from the terms being
mean zero and independent, so they add in quadrature rather than linearly. **That is the strong order
$1/2$.**

For an expectation, the same sum has mean zero, so the dropped terms contribute nothing at leading
order. What remains is the **second** order effect: the dropped term feeds back into later steps
through $X_k$, and the resulting bias is of size $O(h)$ per unit time. **That is the weak order $1$.**

**The general statement**, from the Talay-Tubaro expansion: for a method whose local error has mean
$O(h^{q+1})$ and standard deviation $O(h^{p+1/2})$, the weak order is $q$ and the strong order is $p$.
Euler-Maruyama has $q = 1$ and $p = 1/2$, and Milstein raises $p$ to $1$ while leaving $q$ at $1$.

```python
import numpy as np
from nalib import sde

out = sde.euler_maruyama_is_half_strong_and_one_weak()
print(f"strong, over h = {[f'{v:.4f}' for v in out['strong_steps']]}")
print(f"  errors {[f'{v:.2e}' for v in out['strong_errors']]}")
print(f"  fitted order {out['strong_order']:.4f}, against 1/2")
print(f"weak, over h = {[f'{v:.4f}' for v in out['weak_steps']]}")
print(f"  errors {[f'{v:.2e}' for v in out['weak_errors']]}")
print(f"  fitted order {out['weak_order']:.4f}, against 1")

print("\nthe strong errors halve every four steps and the weak ones every two,")
print("which is the two rates side by side:")
strong = np.asarray(out["strong_errors"])
weak = np.asarray(out["weak_errors"])
print(f"  strong ratios: {[f'{v:.3f}' for v in strong[:-1] / strong[1:]]}, "
      f"predicted {np.sqrt(2.0):.3f}")
print(f"  weak ratios:   {[f'{v:.3f}' for v in weak[:-1] / weak[1:]]}, predicted 2.000")

assert out["strong_is_a_half"]
assert out["weak_is_one"]
```

The strong errors fall by $\sqrt2$ per halving of the step and the weak errors by $2$, which is the
two orders read off the ratios rather than from a fitted line.

### 3.1 A derivative free strong order one method

Milstein needs $b'$. A stochastic Runge-Kutta method replaces it with a difference, exactly as
Runge-Kutta replaced the Jacobian in lesson 68:

$$
\tilde X = X_k + a\,h + b\,\sqrt h , \qquad
X_{k+1} = X_k + a\,h + b\,\Delta W
+ \frac{b(\tilde X) - b(X_k)}{2\sqrt h}\big(\Delta W^{2} - h\big) .
$$

The difference quotient approximates $b\,b'$ to the accuracy the extra term needs.

```python
import math

import numpy as np
from nalib import sde


def runge_kutta(problem, steps, horizon=1.0, increments=None, paths=1, seed=42):
    """Strong order 1 without b', by a supporting value in place of the derivative."""
    h = float(horizon) / steps
    if increments is None:
        increments = math.sqrt(h) * np.random.default_rng(seed).standard_normal((paths, steps))
    x = np.full(increments.shape[0], float(problem["start"]))
    t = 0.0
    for k in range(steps):
        step = increments[:, k]
        drift, diffusion = problem["a"](t, x), problem["b"](t, x)
        helper = x + drift * h + diffusion * math.sqrt(h)
        x = (x + drift * h + diffusion * step
             + 0.5 * (problem["b"](t, helper) - diffusion) * (step * step - h) / math.sqrt(h))
        t += h
    return {"x": x}


problem = sde.geometric_brownian()
powers, paths = (4, 5, 6, 7, 8), 40000
finest = 2 ** max(powers)
fine = sde.brownian(finest, horizon=1.0, paths=paths, seed=42)
truth = problem["exact"](1.0, fine["values"][:, -1])

print(f"{'h':>10}{'euler-maruyama':>18}{'milstein':>14}{'derivative free':>18}")
errors = {"euler-maruyama": [], "milstein": [], "derivative free": []}
steps_used = []
for power in powers:
    n = 2 ** power
    coarse = sde.coarsen(fine["increments"], finest // n)
    row = {
        "euler-maruyama": sde.euler_maruyama(problem, n, increments=coarse)["x"],
        "milstein": sde.milstein(problem, n, increments=coarse)["x"],
        "derivative free": runge_kutta(problem, n, increments=coarse)["x"],
    }
    steps_used.append(1.0 / n)
    for name, values in row.items():
        errors[name].append(float(np.mean(np.abs(values - truth))))
    print(f"{1.0 / n:>10.5f}{errors['euler-maruyama'][-1]:>18.4e}"
          f"{errors['milstein'][-1]:>14.4e}{errors['derivative free'][-1]:>18.4e}")

logs = np.log(steps_used)
print()
for name, values in errors.items():
    print(f"  {name:>18}: fitted strong order "
          f"{float(np.polyfit(logs, np.log(values), 1)[0]):+.4f}")
```

**The derivative free method reaches strong order $1.0058$**, matching Milstein's $0.9911$, and its
errors are only $21$ to $26$ per cent larger at every step size.

That is the same trade Runge-Kutta made in Part 10 and for the same reason: **an extra function
evaluation is almost always cheaper than a derivative**, and it is available when the derivative is
not, which for a black box diffusion coefficient is the usual situation.

The supporting value uses $\sqrt h$ rather than $\Delta W$, and that matters. Using the actual
increment would make the two evaluations correlated in a way that spoils the cancellation, and the
method would drop back to strong order $1/2$.

### 3.2 Multilevel Monte Carlo

The estimator telescopes across a hierarchy of step sizes:

$$
\mathbb{E}[P_L] = \mathbb{E}[P_0] + \sum_{\ell=1}^{L}\mathbb{E}[P_\ell - P_{\ell-1}] ,
$$

where $P_\ell$ is the payoff computed with $2^{\ell}$ steps. Each difference is estimated with its own
number of paths, and the crucial point is that $P_\ell$ and $P_{\ell-1}$ are computed **on the same
Brownian path**, so their difference is small and its variance falls with $\ell$.

```python
import math

import numpy as np
from nalib import sde

problem = sde.geometric_brownian()
payoff = lambda x: np.maximum(x - 1.0, 0.0)
exact = (sde.black_scholes(1.0, 1.0, problem["drift"], problem["volatility"], 1.0)
         * math.exp(problem["drift"]))
print(f"the target is E[max(X_1 - 1, 0)] = {exact:.8f}")


def level_difference(level, paths, seed):
    """The fine payoff minus the coarse one, on the same Brownian paths."""
    n = 2 ** level
    rng = np.random.default_rng(int(seed))
    fine = math.sqrt(1.0 / n) * rng.standard_normal((paths, n))
    high = payoff(sde.euler_maruyama(problem, n, increments=fine)["x"])
    if level == 0:
        return high
    low = payoff(sde.euler_maruyama(problem, n // 2, increments=sde.coarsen(fine, 2))["x"])
    return high - low


print(f"\n{'level':>7}{'steps':>8}{'mean':>14}{'variance':>14}{'variance ratio':>17}")
previous = None
levels_shown = 9
for level in range(levels_shown):
    values = level_difference(level, 40000, 100 + level)
    spread = float(np.var(values, ddof=1))
    ratio = "-" if previous is None else f"{previous / spread:.3f}"
    print(f"{level:>7}{2 ** level:>8}{float(np.mean(values)):>14.3e}{spread:>14.3e}"
          f"{ratio:>17}")
    previous = spread
```

**Both columns fall geometrically, and they fall for different reasons.**

The **mean** falls like $2^{-\ell}$, which is the weak order: $\mathbb{E}[P_\ell - P_{\ell-1}]$ is
the difference of two biases of size $O(h)$.

The **variance** falls like $2^{-\ell}$ too, and this is the strong order at work:
$\operatorname{Var}(P_\ell - P_{\ell-1}) \le \mathbb{E}[(P_\ell - P_{\ell-1})^{2}] = O(h^{2p}) = O(h)$
for $p = 1/2$. From level $2$ onward the ratios are close to $2$.

That variance decay is the whole mechanism, and exercise 5.1 develops why.

```python
import math

import numpy as np
from nalib import sde

print(f"{'target':>9}{'plain steps':>13}{'plain paths':>13}{'plain cost':>13}"
      f"{'levels':>8}{'MLMC cost':>12}{'saving':>9}{'MLMC error':>13}")
for target in (2e-3, 1e-3, 5e-4, 2.5e-4):
    share = target / math.sqrt(2.0)              # split the budget between bias and noise
    steps = int(math.ceil(0.03 / share))
    paths = int(math.ceil(2.0 * 0.09 / share ** 2))
    plain_cost = steps * paths

    levels = max(int(math.ceil(math.log2(0.03 / share))), 2)
    spread = {l: float(np.var(level_difference(l, 8000, 900 + l), ddof=1))
              for l in range(levels + 1)}
    cost = {l: 2 ** l for l in range(levels + 1)}
    total = sum(math.sqrt(spread[l] * cost[l]) for l in range(levels + 1))
    counts = {l: max(int(math.ceil(2.0 * total * math.sqrt(spread[l] / cost[l])
                                   / share ** 2)), 100)
              for l in range(levels + 1)}
    mlmc_cost = sum(counts[l] * cost[l] for l in range(levels + 1))
    estimate = sum(float(np.mean(level_difference(l, counts[l], 500 + l)))
                   for l in range(levels + 1))
    print(f"{target:>9.1e}{steps:>13}{paths:>13}{plain_cost:>13}{levels:>8}"
          f"{mlmc_cost:>12}{plain_cost / mlmc_cost:>9.1f}{abs(estimate - exact):>13.2e}")
```

**The saving grows as the target tightens**, from $7.7$ to $31.2$ as the accuracy goes from
$2\times10^{-3}$ to $2.5\times10^{-4}$, and every run lands at or inside its target.

The growth is the point. Plain simulation costs $O(\varepsilon^{-3})$: $\varepsilon^{-1}$ steps to
control the bias times $\varepsilon^{-2}$ paths to control the noise. Multilevel costs
$O(\varepsilon^{-2}(\log\varepsilon)^{2})$ when the variance decays as fast as the cost grows, which
is exactly the case here. So the saving is $O(\varepsilon^{-1})$ up to logarithms, and each halving of
$\varepsilon$ should roughly double it: the measured ratios are $1.53$, $1.60$ and $1.65$, short of
$2$ by the logarithm.

### 3.3 An implicit method, and a stiff SDE

The drift can be made implicit exactly as in lesson 74. The diffusion cannot: making $b$ implicit
would put $\Delta W$ inside a solve and destroy the martingale property, so every standard scheme
keeps the diffusion explicit.

$$
X_{k+1} = X_k + a(X_{k+1})\,h + b(X_k)\,\Delta W .
$$

```python
import math

import numpy as np
from nalib import sde


def stiff(rate=200.0, volatility=0.3):
    """dX = -rate X dt + volatility dW: mean reverting fast, with additive noise."""
    return {"a": lambda t, x: -rate * np.asarray(x, dtype=float),
            "b": lambda t, x: volatility * np.ones_like(np.asarray(x, dtype=float)),
            "b_prime": lambda t, x: np.zeros_like(np.asarray(x, dtype=float)),
            "exact": None, "start": 1.0, "rate": rate, "volatility": volatility,
            "mean": lambda t: math.exp(-rate * float(t)),
            "variance": lambda t: (volatility ** 2 / (2.0 * rate)
                                   * (1.0 - math.exp(-2.0 * rate * float(t))))}


def implicit_euler(problem, steps, horizon=1.0, increments=None, paths=1, seed=42):
    """The drift implicit, the diffusion explicit, which is the only stable arrangement."""
    h = float(horizon) / steps
    if increments is None:
        increments = math.sqrt(h) * np.random.default_rng(seed).standard_normal((paths, steps))
    x = np.full(increments.shape[0], float(problem["start"]))
    for k in range(steps):
        # for a linear drift the solve is exact: x_new (1 + rate h) = x + b dW
        x = (x + problem["b"](0.0, x) * increments[:, k]) / (1.0 + problem["rate"] * h)
    return {"x": x}


problem = stiff()
print(f"the drift rate is {problem['rate']:g}, so explicit stability needs "
      f"h < 2/rate = {2.0 / problem['rate']:.5f}")
print(f"the transient exp(-rate t) is {problem['mean'](1.0):.1e} by time 1, so the answer "
      f"is the")
print(f"stationary distribution: mean 0 and standard deviation "
      f"{math.sqrt(problem['variance'](1.0)):.5f}")
print(f"\n{'steps':>8}{'h':>10}{'h times rate':>15}{'explicit mean':>16}"
      f"{'explicit spread':>18}{'implicit spread':>18}")
for steps in (50, 100, 150, 400, 2000):
    h = 1.0 / steps
    rng = np.random.default_rng(7)
    increments = math.sqrt(h) * rng.standard_normal((4000, steps))
    with np.errstate(over="ignore", invalid="ignore"):
        explicit = sde.euler_maruyama(problem, steps, increments=increments)["x"]
    implicit = implicit_euler(problem, steps, increments=increments)["x"]
    finite = np.all(np.isfinite(explicit))
    print(f"{steps:>8}{h:>10.5f}{h * problem['rate']:>15.3f}"
          f"{(f'{float(np.mean(explicit)):.3e}' if finite else 'overflow'):>16}"
          f"{(f'{float(np.std(explicit, ddof=1)):.3e}' if finite else '-'):>18}"
          f"{float(np.std(implicit, ddof=1)):>18.3e}")

print(f"\nand the stationary variance, which is what the noise sets:")
for steps in (100, 400, 2000):
    h = 1.0 / steps
    rng = np.random.default_rng(7)
    increments = math.sqrt(h) * rng.standard_normal((20000, steps))
    implicit = implicit_euler(problem, steps, increments=increments)["x"]
    print(f"  {steps:>5} steps: variance {float(np.var(implicit, ddof=1)):.6f} "
          f"against {problem['variance'](1.0):.6f}")
```

**The explicit method diverges the moment $h\,\text{rate}$ exceeds $2$.** At $h\,\text{rate} = 4$
its spread is $10^{23}$, and at exactly $2$ it sits on the stability boundary: the amplification
factor is $|1-2| = 1$, so the scheme neither grows nor decays and the spread stays at its initial
size instead of settling to the stationary one. Below $2$ it works. That is lesson 74's condition
unchanged.

But the spread columns show a price lesson 74 did not have. **The implicit method is stable at every
step and it gets the stationary spread wrong at a coarse one**, because damping the drift also damps
the noise the drift is supposed to balance. The stationary variance is $\sigma^{2}/2\theta$, and the
implicit scheme reaches $0.000113$, $0.000178$ and $0.000215$ of a true $0.000225$ at $100$, $400$
and $2000$ steps: it converges, from below, only as the step refines.

That is the characteristic difficulty of stiff SDEs: **stability and the correct invariant
distribution are two different requirements**, and a scheme can have the first without the second.
The methods designed for this, the stochastic theta methods with $\theta = 1/2$ and the balanced
implicit methods, exist precisely to get both.

### 3.4 Asian and barrier options

Both depend on the **whole path**, not just its endpoint, which changes which order matters.

```python
import math

import numpy as np
from nalib import sde

problem = sde.geometric_brownian(drift=0.05, volatility=0.3)
paths, horizon = 40000, 1.0


def price(method, steps, kind, seed=42):
    rng = np.random.default_rng(seed)
    h = horizon / steps
    increments = math.sqrt(h) * rng.standard_normal((paths, steps))
    run = method(problem, steps, horizon=horizon, increments=increments)
    trail = run["path"]
    if kind == "european":
        payoff = np.maximum(trail[:, -1] - 1.0, 0.0)
    elif kind == "asian":
        payoff = np.maximum(trail.mean(axis=1) - 1.0, 0.0)
    else:
        alive = trail.max(axis=1) < 1.5
        payoff = np.where(alive, np.maximum(trail[:, -1] - 1.0, 0.0), 0.0)
    return math.exp(-problem["drift"] * horizon) * float(np.mean(payoff))


print(f"{'option':>12}{'steps':>8}{'Euler-Maruyama':>18}{'Milstein':>14}"
      f"{'difference':>14}")
for kind in ("european", "asian", "barrier"):
    reference = price(sde.milstein, 2048, kind)
    for steps in (16, 64, 256):
        plain = price(sde.euler_maruyama, steps, kind)
        better = price(sde.milstein, steps, kind)
        print(f"{kind:>12}{steps:>8}{plain:>18.6f}{better:>14.6f}"
              f"{abs(plain - better):>14.2e}")
    print(f"{'':>12}{'2048':>8}{'':>18}{reference:>14.6f}   reference")
```

**For the European option the two methods agree**, as section 5 of the lesson found: the payoff
depends only on the endpoint, which is a weak quantity.

**For the Asian option they agree too.** The average over the path is still a linear functional of the
path, so its expectation is a weak quantity and the weak orders being equal is what matters.

**For the barrier option the two methods also agree**, to $10^{-4}$ at $256$ steps, and that is not
the interesting column. **The interesting one is how far the barrier price moves with the step count
at all.**

From $16$ to $2048$ steps the European price moves by $2$ per cent and the Asian by $3$, both
converging quickly. The barrier price moves from $0.0633$ to $0.0526$, **twenty per cent**, and is
still drifting at $256$ steps.

The reason is a bias neither method addresses. **The discrete maximum over grid points systematically
underestimates the continuous maximum**, because the path can cross the barrier and return between
two grid times, and a discrete monitor never sees it. So the simulated option survives more often than
it should and is priced too high, by an amount that falls only like $O(\sqrt h)$: far slower than
either method's order, which is why refining the step is such a poor way to fix it.

The standard repair is a Brownian bridge correction, which computes the probability of an unobserved
crossing between two grid points analytically and kills the path with that probability. It restores
$O(h)$ convergence and costs one extra exponential per step.

So the answer to which method matters here is: neither. **The dominant error is in the payoff's
discretization rather than the path's**, and choosing between Euler-Maruyama and Milstein is choosing
between two errors that are both an order of magnitude below the one that matters.

### 3.5 The Heston model and the negative variance problem

$$
dS = \mu S\,dt + \sqrt{v}\,S\,dW_1 , \qquad
dv = \kappa(\theta - v)\,dt + \xi\sqrt{v}\,dW_2 ,
$$

with $dW_1\,dW_2 = \rho\,dt$. The variance process is a square root diffusion, and its exact solution
stays positive when the Feller condition $2\kappa\theta \ge \xi^{2}$ holds. **The discretization does
not.**

```python
import math

import numpy as np

kappa, theta, xi, rho = 2.0, 0.04, 0.5, -0.7
spot, drift, horizon = 100.0, 0.03, 1.0
feller = 2.0 * kappa * theta - xi ** 2
print(f"the Feller condition 2 kappa theta - xi**2 = {feller:+.4f}, "
      f"so the exact variance {'stays' if feller >= 0 else 'can reach'} positive")


def heston(steps, paths, fix, seed=42):
    rng = np.random.default_rng(int(seed))
    h = horizon / steps
    first = math.sqrt(h) * rng.standard_normal((paths, steps))
    second = rho * first + math.sqrt(1.0 - rho ** 2) * math.sqrt(h) * rng.standard_normal(
        (paths, steps))
    s = np.full(paths, spot)
    v = np.full(paths, theta)
    negatives = 0
    for k in range(steps):
        raw = v + kappa * (theta - v) * h + xi * np.sqrt(np.maximum(v, 0.0)) * second[:, k]
        negatives += int(np.count_nonzero(raw < 0.0))
        if fix == "absorb":
            v = np.maximum(raw, 0.0)
        elif fix == "reflect":
            v = np.abs(raw)
        else:                                    # keep the drift, drop the noise below zero
            v = np.where(raw < 0.0, v + kappa * (theta - v) * h, raw)
        s = s * np.exp((drift - 0.5 * v) * h + np.sqrt(np.maximum(v, 0.0)) * first[:, k])
    return s, v, negatives


print(f"\n{'fix':>10}{'steps':>8}{'negatives seen':>17}{'mean variance':>16}"
      f"{'wanted':>10}{'mean spot':>13}")
for fix in ("absorb", "reflect", "drift only"):
    for steps in (50, 400):
        s, v, negatives = heston(steps, 20000, fix)
        print(f"{fix:>10}{steps:>8}{negatives:>17}{float(np.mean(v)):>16.6f}"
              f"{theta:>10.4f}{float(np.mean(s)):>13.4f}")
print(f"  the spot should have mean {spot * math.exp(drift * horizon):.4f}")
```

All three repairs keep the simulation running and they give **different answers**, which is the point.

**Absorbing** at zero is the cheapest and biases the variance upward least at fine steps, but it
introduces an atom at zero that the true process does not have.

**Reflecting** keeps the process away from zero and biases the variance **upward**, because a
reflection turns a small negative into a small positive rather than into zero.

**Dropping the noise** when the step would go negative is the mildest of the three and is closest to
the exact process at fine steps.

The number of negative excursions falls sharply with the step count, so all three converge to the
same answer; the disagreement at $50$ steps is the discretization error made visible. The methods
that avoid the problem entirely, the exact scheme of Broadie and Kaya and the quadratic exponential
scheme of Andersen, sample the variance from its true transition distribution and never discretize
it.

### 4.1 The balance between steps and paths

At a fixed budget $B$ of increments, choosing $n$ steps leaves $B/n$ paths. The bias falls like $c/n$
and the noise like $\sigma\sqrt{n/B}$, so the total error is

$$
\sqrt{\frac{c^{2}}{n^{2}} + \frac{\sigma^{2}n}{B}} ,
$$

minimized at $n \propto B^{1/3}$.

```python
import numpy as np
from nalib import sde

problem = sde.geometric_brownian()
target = problem["mean"](1.0)
repeats = 12
for budget in (2 ** 20, 2 ** 22):
    print(f"\nbudget {budget} increments")
    print(f"{'steps':>8}{'paths':>10}{'bias':>12}{'noise':>12}{'total rms':>13}")
    best = None
    for power in range(1, 12):
        steps = 2 ** power
        paths = budget // steps
        if paths < 200:
            continue
        runs = np.array([float(np.mean(sde.euler_maruyama(problem, steps, paths=paths,
                                                          seed=s)["x"]))
                         for s in range(repeats)])
        rms = float(np.sqrt(np.mean((runs - target) ** 2)))
        print(f"{steps:>8}{paths:>10}"
              f"{abs(float(np.mean(runs)) - target):>12.3e}"
              f"{float(np.std(runs, ddof=1)):>12.3e}{rms:>13.3e}")
        if best is None or rms < best[0]:
            best = (rms, steps)
    print(f"  the smallest total is at {best[1]} steps")
```

**The optimum is at eight steps for both budgets**, and the two columns show why: at two steps the
bias is $2.5\times10^{-3}$ and the noise $3.4\times10^{-4}$, at $2048$ steps the bias is
$4.0\times10^{-3}$ and the noise $1.2\times10^{-2}$, and they cross around eight.

Two things are worth reading off it.

**The optimum is very coarse.** Eight time steps is a crude discretization by any standard, and it is
the right choice because the weak bias falls like $1/n$ while the noise falls only like $\sqrt{n/B}$:
buying accuracy through steps is cheap and buying it through paths is expensive.

**And the optimum is very flat.** At a budget of $2^{22}$ the totals at $8$ and $16$ steps are
$8.37\times10^{-4}$ and $8.56\times10^{-4}$, within two per cent, so the exact choice hardly matters.
The $B^{1/3}$ prediction says quadrupling the budget should move the optimum by $1.59$, which is a
move from $8$ to $13$: consistent with the near tie between $8$ and $16$ at the larger budget and too
small to resolve on a doubling grid.

The practical rule is that **a fine time grid is usually wasted effort in a weak calculation**, which
is the same conclusion exercise 3.2 reaches by a different route.

### 4.2 Both orders with mixed noise

```python
import math

import numpy as np
from nalib import sde


def mixed(drift=0.1, additive=0.2, multiplicative=0.3, start=1.0):
    """dX = mu X dt + (sigma_0 + sigma_1 X) dW, so b' is a nonzero constant."""
    return {"a": lambda t, x: drift * np.asarray(x, dtype=float),
            "b": lambda t, x: additive + multiplicative * np.asarray(x, dtype=float),
            "b_prime": lambda t, x: multiplicative * np.ones_like(np.asarray(x, dtype=float)),
            "exact": None, "start": start, "drift": drift,
            "additive": additive, "multiplicative": multiplicative}


problem = mixed()
powers, paths = (4, 5, 6, 7, 8), 40000
finest = 2 ** max(powers)
fine = sde.brownian(finest, horizon=1.0, paths=paths, seed=42)
# no closed form here, so the reference is the finest Milstein run on the same paths
reference = sde.milstein(problem, finest, increments=fine["increments"])["x"]
reference_mean = float(np.mean(reference))

print(f"{'h':>10}{'EM strong':>14}{'Milstein strong':>18}{'EM weak':>13}"
      f"{'Milstein weak':>16}")
rows = {"em_s": [], "mil_s": [], "em_w": [], "mil_w": []}
steps_used = []
for power in powers[:-1]:
    n = 2 ** power
    coarse = sde.coarsen(fine["increments"], finest // n)
    plain = sde.euler_maruyama(problem, n, increments=coarse)["x"]
    better = sde.milstein(problem, n, increments=coarse)["x"]
    rows["em_s"].append(float(np.mean(np.abs(plain - reference))))
    rows["mil_s"].append(float(np.mean(np.abs(better - reference))))
    rows["em_w"].append(abs(float(np.mean(plain)) - reference_mean))
    rows["mil_w"].append(abs(float(np.mean(better)) - reference_mean))
    steps_used.append(1.0 / n)
    print(f"{1.0 / n:>10.5f}{rows['em_s'][-1]:>14.3e}{rows['mil_s'][-1]:>18.3e}"
          f"{rows['em_w'][-1]:>13.3e}{rows['mil_w'][-1]:>16.3e}")

logs = np.log(steps_used)
print()
for key, label in (("em_s", "Euler-Maruyama strong"), ("mil_s", "Milstein strong"),
                   ("em_w", "Euler-Maruyama weak"), ("mil_w", "Milstein weak")):
    print(f"  {label:>24}: {float(np.polyfit(logs, np.log(rows[key]), 1)[0]):+.4f}")
```

**The orders are what they were with purely multiplicative noise.** Adding a constant to the diffusion
does not change $b'$, which stays equal to the multiplicative coefficient, so Milstein's extra term is
unchanged and so is everything that follows from it.

The measurement is worth having because it separates two things that are easy to confuse. It is
**not** the presence of additive noise that makes Milstein equal to Euler-Maruyama, as the lesson's
Ornstein-Uhlenbeck control might suggest. It is $b'$ being zero. Additive noise is one way to make
$b'$ zero and mixed noise is not, so the two methods stay different here.

The reference is the finest Milstein run rather than a closed form, because this problem has none.
That makes the finest step's error zero by construction, which is why the table stops one step short
of it.

### 4.3 The weak error against the payoff

```python
import numpy as np
from nalib import sde

problem = sde.geometric_brownian()
powers, paths = (1, 2, 3, 4, 5), 400000
finest = 2 ** max(powers)
fine = sde.brownian(finest, horizon=1.0, paths=paths, seed=42)

print(f"{'payoff':>24}{'fitted weak order':>20}{'error at h = 1/2':>20}")
for name, g in (("smooth, x", lambda x: x),
                ("smooth, x**2", lambda x: x * x),
                ("a kink, max(x-1,0)", lambda x: np.maximum(x - 1.0, 0.0)),
                ("a jump, the indicator", lambda x: (x > 1.0).astype(float))):
    truth = float(np.mean(g(problem["exact"](1.0, fine["values"][:, -1]))))
    errors, steps_used = [], []
    for power in powers:
        n = 2 ** power
        run = sde.euler_maruyama(problem, n,
                                 increments=sde.coarsen(fine["increments"], finest // n))
        errors.append(abs(float(np.mean(g(run["x"]))) - truth))
        steps_used.append(1.0 / n)
    order = float(np.polyfit(np.log(steps_used), np.log(errors), 1)[0])
    print(f"{name:>24}{order:>20.4f}{errors[0]:>20.3e}")
```

**The weak order is one for all four payoffs**, including the discontinuous indicator: $1.013$,
$0.981$, $1.124$ and $1.005$.

That contradicts the usual statement of the theorem, which asks for a smooth $g$, and the reason is
worth understanding. The theorem's hypothesis is sufficient, not necessary. Here the **law** of
$X(T)$ has a smooth density, and

$$
\mathbb{E}g(X_N) = \int g(x)\,p_N(x)\,dx ,
$$

so the roughness of $g$ is integrated against a smooth density and does not survive. What would break
the order is a rough **density**, which happens for a degenerate diffusion or a payoff evaluated at a
stopping time rather than a fixed time.

**What does change is the constant.** At $h = 1/2$ the errors are $2.6\times10^{-3}$ for the identity
and $2.9\times10^{-2}$ for the indicator, a factor of $11$. So the discontinuous payoff needs about
eleven times the steps for the same weak accuracy, at the same order.

That distinction is practically important. It means a digital option can be priced with a standard
scheme and simply needs a finer grid, rather than needing a different method, which is not what the
smoothness hypothesis would lead one to expect.

### 5.1 Why multilevel works

**The estimator.**

$$
\hat Y = \frac{1}{N_0}\sum P_0^{(i)}
+ \sum_{\ell=1}^{L}\frac{1}{N_\ell}\sum\big(P_\ell^{(i)} - P_{\ell-1}^{(i)}\big) ,
$$

with independent samples at each level and the two payoffs at level $\ell$ computed on the **same**
Brownian path.

**The cost and the variance.** Level $\ell$ costs $C_\ell = O(2^{\ell})$ per path and has variance
$V_\ell$. Minimizing the total cost $\sum N_\ell C_\ell$ subject to a total variance
$\sum V_\ell/N_\ell = \varepsilon^{2}/2$ by Lagrange multipliers gives

$$
N_\ell \propto \sqrt{V_\ell / C_\ell} ,
\qquad
\text{total cost} \propto \frac{1}{\varepsilon^{2}}\left(\sum_\ell \sqrt{V_\ell C_\ell}\right)^{2} .
$$

**Where the strong order enters.** $V_\ell = \operatorname{Var}(P_\ell - P_{\ell-1})$, and for a
Lipschitz payoff

$$
V_\ell \le \mathbb{E}\big[(P_\ell - P_{\ell-1})^{2}\big]
\le K\,\mathbb{E}\big[(X_\ell - X_{\ell-1})^{2}\big] = O(h_\ell^{2p}) ,
$$

with $p$ the **strong** order. So $V_\ell C_\ell = O(2^{\ell(1-2p)})$, and the sum
$\sum\sqrt{V_\ell C_\ell}$ converges when $2p > 1$, is logarithmic when $2p = 1$, and diverges when
$2p < 1$.

**That is the answer to the exercise.** The quantity wanted is an expectation, a weak quantity, and
the weak order sets only how many levels are needed, $L = O(\log(1/\varepsilon))$. But the **cost** is
set by how fast the level variances decay, and that is the strong order. Euler-Maruyama's $p = 1/2$
sits exactly at the boundary, giving the $O(\varepsilon^{-2}(\log\varepsilon)^{2})$ that exercise 3.2
measures, and Milstein's $p = 1$ would give a clean $O(\varepsilon^{-2})$.

```python
import math

import numpy as np
from nalib import sde

levels_shown = 9
print("the level variances, and the sum that decides the cost:")
print(f"{'level':>7}{'V':>12}{'C':>8}{'V C':>12}{'sqrt(V C)':>13}{'running sum':>14}")
running = 0.0
for level in range(levels_shown):
    spread = float(np.var(level_difference(level, 40000, 100 + level), ddof=1))
    cost = 2 ** level
    running += math.sqrt(spread * cost)
    print(f"{level:>7}{spread:>12.3e}{cost:>8}{spread * cost:>12.3e}"
          f"{math.sqrt(spread * cost):>13.4f}{running:>14.4f}")
print("\nthe terms are nearly constant, which is the 2p = 1 boundary case:")
print("a convergent sum would give cost O(eps**-2) and a divergent one O(eps**-3)")
```

The terms $\sqrt{V_\ell C_\ell}$ are nearly constant from level two onward, which is exactly the
borderline: $V_\ell \propto 2^{-\ell}$ against $C_\ell \propto 2^{\ell}$. The running sum therefore
grows like $L = O(\log(1/\varepsilon))$, and squaring it gives the $(\log\varepsilon)^{2}$ in the cost.

**Milstein would remove that logarithm**, which is the one setting where its extra term is worth
paying for even though the target is an expectation. That is the answer to the puzzle the lesson
raised: Milstein buys nothing weakly on its own, and buys a logarithm inside a multilevel estimator.

### 5.2 Ito against Stratonovich

**The difference is where the integrand is evaluated.** Both define $\int b(X)\,dW$ as a limit of
sums, and they differ in the sample point:

$$
\text{Ito: } \sum b(X_{t_k})\,\Delta W_k , \qquad
\text{Stratonovich: } \sum b\!\left(\tfrac{X_{t_k}+X_{t_{k+1}}}{2}\right)\Delta W_k .
$$

For an ordinary integral the choice does not matter, because the integrand's variation over a step is
$O(h)$ and there are $1/h$ steps. Here the increments are $O(\sqrt h)$ and the two choices differ by
a term that does **not** vanish.

**The conversion.** Writing the Stratonovich integral as $\int b \circ dW$,

$$
\int_0^T b(X)\circ dW = \int_0^T b(X)\,dW + \tfrac12\int_0^T b(X)\,b'(X)\,dt ,
$$

so a Stratonovich equation $dX = \tilde a\,dt + b\circ dW$ is the Ito equation

$$
dX = \left(\tilde a + \tfrac12 b\,b'\right)dt + b\,dW .
$$

**The half $b b'$ is the same quantity that appears in Milstein's term**, which is not a coincidence:
both come from the correlation between the integrand and the increment over a step.

**Which one a physical model should use.** The answer is decided by where the noise came from.

*Use Stratonovich* when the noise is an idealization of a real process with a short but non-zero
correlation time. The Wong-Zakai theorem says that if the driving noise is smooth and its correlation
time is sent to zero, the limit is the **Stratonovich** equation. Physical noise is always smooth at
some scale, so this is the usual case in physics and engineering.

*Use Ito* when the noise is genuinely a sequence of independent shocks, so that the coefficient at
each instant is determined before the shock arrives. That is the case in finance, where a trading
decision is made on the information available and cannot anticipate the next price move, and it is
why the martingale property, which only Ito has, is the one that matters there.

**And the two are not interchangeable in numerical work.** A scheme written for one and applied to
the other silently solves a different equation, differing in drift by $\tfrac12 b b'$.

```python
import math

import numpy as np
from nalib import sde

drift, volatility, horizon = 0.1, 0.5, 1.0
same = sde.geometric_brownian(drift=drift, volatility=volatility)
print(f"the Ito equation dX = {drift} X dt + {volatility} X dW")
print(f"as a Stratonovich equation its drift is "
      f"{drift} - (1/2)({volatility})**2 = "
      f"{drift - 0.5 * volatility ** 2:.4f}")

steps, paths = 4000, 40000
h = horizon / steps
rng = np.random.default_rng(42)
increments = math.sqrt(h) * rng.standard_normal((paths, steps))

ito = sde.euler_maruyama(same, steps, increments=increments)["x"]
# the midpoint rule, which converges to the Stratonovich solution
x = np.full(paths, same["start"])
for k in range(steps):
    predictor = x + same["a"](0.0, x) * h + same["b"](0.0, x) * increments[:, k]
    middle = 0.5 * (x + predictor)
    x = x + same["a"](0.0, middle) * h + same["b"](0.0, middle) * increments[:, k]

print(f"\n{'scheme':>26}{'mean':>14}{'predicted':>14}")
print(f"{'left endpoint, Ito':>26}{float(np.mean(ito)):>14.6f}"
      f"{same['mean'](horizon):>14.6f}")
print(f"{'midpoint, Stratonovich':>26}{float(np.mean(x)):>14.6f}"
      f"{math.exp((drift + 0.5 * volatility ** 2) * horizon):>14.6f}")
print(f"\nthe two schemes differ by a factor of "
      f"{float(np.mean(x)) / float(np.mean(ito)):.4f}, and the predicted factor is "
      f"{math.exp(0.5 * volatility ** 2 * horizon):.4f}")
```

The same increments through the same equation give two different answers depending on the sample
point, and their ratio is the predicted $e^{\sigma^{2}T/2}$.

### 5.3 The discretization bias is not a rounding error

**In Part 10 the step could be refined until rounding stopped it.** The error of a $p$-th order method
was $C h^{p} + O(\varepsilon/h)$, the two terms crossed at $h \approx \varepsilon^{1/(p+1)}$, and
lesson 70 measured that crossing. The floor was machine precision and nothing else.

**Here there is a second floor, and it is enormously higher.** Estimating an expectation from $M$
paths has a Monte Carlo error of $O(M^{-1/2})$ that the step size does not touch. So the total error
is

$$
\underbrace{C\,h^{q}}_{\text{discretization bias}} + \underbrace{\frac{\sigma}{\sqrt M}}_{\text{sampling}} ,
$$

and refining $h$ below the point where the first term falls under the second buys **nothing at all**.
Exercise 4.1 measures that point at eight steps for a budget of a million increments.

**Three consequences follow, and they change how the calculation is organized.**

*The step size is a budget decision, not an accuracy decision.* Halving $h$ doubles the cost per path
and therefore halves the number of paths at a fixed budget, which multiplies the sampling error by
$\sqrt2$. The bias and the noise trade against each other and the optimum is the $B^{1/3}$ of 4.1.

*Convergence cannot be checked by refining alone.* In Part 10 a run at $h$ and a run at $h/2$ whose
answers agree is evidence of convergence. Here two such runs agree whenever both are inside their
common sampling noise, which says nothing about the bias. The bias has to be measured separately, on
**paired** runs sharing the same Brownian path, which is what every order measurement in this lesson
does and why the coarsening operation exists.

*And rounding never enters.* The $\sqrt\varepsilon$ barrier of lesson 84 and the $\varepsilon/h$ term
of lesson 70 are both far below the sampling floor, by a measured factor of $8\times10^{4}$ here and
by $10^{6}$ in lesson 92's exercise 5.1. **Machine precision is irrelevant in a stochastic
calculation**, which is the sharpest possible statement of how different this regime is from every
other part of the course.

```python
import math

import numpy as np
from nalib import sde

problem = sde.geometric_brownian()
target = problem["mean"](1.0)
print("refining the step at a fixed number of paths, and what it buys:")
print(f"{'steps':>8}{'h':>10}{'bias':>13}{'sampling floor':>17}{'total rms':>13}"
      f"{'which dominates':>18}")
paths, repeats = 40000, 20
floor = None
for steps in (2, 8, 32, 128, 512, 2048):
    runs = np.array([float(np.mean(sde.euler_maruyama(problem, steps, paths=paths,
                                                      seed=s)["x"]))
                     for s in range(repeats)])
    bias = abs(float(np.mean(runs)) - target)
    noise = float(np.std(runs, ddof=1))
    if floor is None:
        floor = noise
    rms = float(np.sqrt(np.mean((runs - target) ** 2)))
    print(f"{steps:>8}{1.0 / steps:>10.5f}{bias:>13.3e}{noise:>17.3e}{rms:>13.3e}"
          f"{('bias' if bias > noise else 'sampling'):>18}")
print(f"\nthe rounding floor for this calculation is about "
      f"{np.sqrt(np.finfo(float).eps):.2e}, which is "
      f"{floor / np.sqrt(np.finfo(float).eps):.1e} times smaller than the sampling floor")
```

The table crosses from bias dominated to sampling dominated between eight and thirty two steps, and
past that point refining the step changes the total error not at all. The rounding floor sits about
$8\times10^{4}$ below where the calculation actually stops, so nearly five orders of magnitude of
available precision are never used.

---
